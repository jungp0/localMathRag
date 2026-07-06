using System.Diagnostics;
using System.IO.Compression;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace LocalMathRAG;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
        using var mutex = new Mutex(true, "LocalMathRAG.Launcher.SingleInstance", out var isNew);
        if (!isNew)
        {
            MessageBox.Show("LocalMathRAG is already running.", "LocalMathRAG", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }

        ApplicationConfiguration.Initialize();
        Application.Run(new LauncherContext());
    }
}

internal sealed class LauncherContext : ApplicationContext
{
    private const string Host = "127.0.0.1";
    private const int WebPort = 8765;
    private const int LlamaPort = 8080;
    private const string ModelFileName = "Qwen3-8B-Q4_K_M.gguf";
    private const string ModelUrl = "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf?download=true";

    private readonly NotifyIcon notifyIcon;
    private readonly ToolStripMenuItem startItem;
    private readonly ToolStripMenuItem stopItem;
    private readonly string rootDir;
    private readonly string dataDir;
    private readonly string modelsDir;
    private readonly string runtimeDir;
    private readonly string downloadsDir;
    private Process? webProcess;
    private Process? llamaProcess;
    private bool servicesStarting;

    public LauncherContext()
    {
        rootDir = ResolveRootDir();
        dataDir = Path.Combine(rootDir, "data");
        modelsDir = Path.Combine(dataDir, "models");
        runtimeDir = Path.Combine(dataDir, "runtime");
        downloadsDir = Path.Combine(runtimeDir, "downloads");

        Directory.CreateDirectory(dataDir);
        Directory.CreateDirectory(modelsDir);
        Directory.CreateDirectory(runtimeDir);
        Directory.CreateDirectory(downloadsDir);

        startItem = new ToolStripMenuItem("Start services", null, async (_, _) => await StartServicesAsync(openBrowser: true));
        stopItem = new ToolStripMenuItem("Stop services", null, (_, _) => StopServices());

        notifyIcon = new NotifyIcon
        {
            Text = "LocalMathRAG",
            Icon = SystemIcons.Application,
            Visible = true,
            ContextMenuStrip = new ContextMenuStrip(),
        };
        notifyIcon.ContextMenuStrip.Items.Add(new ToolStripMenuItem("Open WebApp", null, (_, _) => OpenWebApp()));
        notifyIcon.ContextMenuStrip.Items.Add(startItem);
        notifyIcon.ContextMenuStrip.Items.Add(stopItem);
        notifyIcon.ContextMenuStrip.Items.Add(new ToolStripMenuItem("Restart services", null, async (_, _) => await RestartServicesAsync()));
        notifyIcon.ContextMenuStrip.Items.Add(new ToolStripSeparator());
        notifyIcon.ContextMenuStrip.Items.Add(new ToolStripMenuItem("View logs", null, (_, _) => OpenLogs()));
        notifyIcon.ContextMenuStrip.Items.Add(new ToolStripMenuItem("Open data directory", null, (_, _) => OpenPath(dataDir)));
        notifyIcon.ContextMenuStrip.Items.Add(new ToolStripSeparator());
        notifyIcon.ContextMenuStrip.Items.Add(new ToolStripMenuItem("Exit", null, (_, _) => Exit()));
        notifyIcon.DoubleClick += (_, _) => OpenWebApp();

        _ = StartServicesAsync(openBrowser: true);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            notifyIcon.Visible = false;
            notifyIcon.Dispose();
        }
        base.Dispose(disposing);
    }

    private async Task StartServicesAsync(bool openBrowser)
    {
        if (servicesStarting)
        {
            return;
        }

        servicesStarting = true;
        SetMenuState(false);
        try
        {
            ShowStatus("Starting LocalMathRAG...");
            var modelPath = Path.Combine(modelsDir, ModelFileName);
            var llamaServer = await EnsureLlamaServerAsync();
            var activeLlamaServer = llamaServer;
            await EnsureModelAsync(modelPath);

            if (!await IsHttpReadyAsync($"http://{Host}:{LlamaPort}/v1/models", TimeSpan.FromSeconds(2)))
            {
                llamaProcess = StartHidden(
                    llamaServer,
                    Path.GetDirectoryName(llamaServer)!,
                    Path.Combine(dataDir, $"llama-server-{LlamaPort}.out.log"),
                    Path.Combine(dataDir, $"llama-server-{LlamaPort}.err.log"),
                    "-m", modelPath,
                    "--host", Host,
                    "--port", LlamaPort.ToString(),
                    "--ctx-size", "8192",
                    "--n-gpu-layers", "999",
                    "--reasoning", "off");
            }

            if (!await IsHttpReadyAsync($"http://{Host}:{LlamaPort}/v1/models", TimeSpan.FromMinutes(3)))
            {
                ShowStatus("CUDA llama.cpp did not start; trying CPU runtime.");
                llamaProcess?.Kill(entireProcessTree: true);
                var cpuServer = await EnsureLlamaServerAsync("cpu", allowAnyExisting: false);
                activeLlamaServer = cpuServer;
                llamaProcess = StartHidden(
                    cpuServer,
                    Path.GetDirectoryName(cpuServer)!,
                    Path.Combine(dataDir, $"llama-server-{LlamaPort}.out.log"),
                    Path.Combine(dataDir, $"llama-server-{LlamaPort}.err.log"),
                    "-m", modelPath,
                    "--host", Host,
                    "--port", LlamaPort.ToString(),
                    "--ctx-size", "8192",
                    "--reasoning", "off");
            }

            if (!await IsHttpReadyAsync($"http://{Host}:{LlamaPort}/v1/models", TimeSpan.FromMinutes(3)))
            {
                throw new InvalidOperationException("llama.cpp endpoint did not become ready.");
            }

            if (!await IsHttpReadyAsync($"http://{Host}:{WebPort}/api/kbs", TimeSpan.FromSeconds(2)))
            {
                var backendExe = Path.Combine(rootDir, "backend", "lookup-tool-server.exe");
                if (!File.Exists(backendExe))
                {
                    throw new FileNotFoundException("Backend executable was not found.", backendExe);
                }

                webProcess = StartHidden(
                    backendExe,
                    rootDir,
                    Path.Combine(dataDir, $"webapp-{WebPort}.out.log"),
                    Path.Combine(dataDir, $"webapp-{WebPort}.err.log"));
            }

            if (!await IsHttpReadyAsync($"http://{Host}:{WebPort}/api/kbs", TimeSpan.FromMinutes(2)))
            {
                throw new InvalidOperationException("WebApp endpoint did not become ready.");
            }

            await ConfigureModelAsync(modelPath, activeLlamaServer);
            ShowStatus("LocalMathRAG is ready.");
            if (openBrowser)
            {
                OpenWebApp();
            }
        }
        catch (Exception ex)
        {
            File.AppendAllText(Path.Combine(dataDir, "launcher.err.log"), DateTimeOffset.Now + " " + ex + Environment.NewLine, Encoding.UTF8);
            MessageBox.Show(ex.Message, "LocalMathRAG startup failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            servicesStarting = false;
            SetMenuState(true);
        }
    }

    private async Task RestartServicesAsync()
    {
        StopServices();
        await Task.Delay(1000);
        await StartServicesAsync(openBrowser: true);
    }

    private void StopServices()
    {
        TryKill(webProcess);
        TryKill(llamaProcess);
        webProcess = null;
        llamaProcess = null;
        ShowStatus("LocalMathRAG services stopped.");
    }

    private void Exit()
    {
        StopServices();
        notifyIcon.Visible = false;
        Application.Exit();
    }

    private void SetMenuState(bool enabled)
    {
        startItem.Enabled = enabled;
        stopItem.Enabled = enabled;
    }

    private static void TryKill(Process? process)
    {
        try
        {
            if (process is { HasExited: false })
            {
                process.Kill(entireProcessTree: true);
            }
        }
        catch
        {
        }
    }

    private async Task<string> EnsureLlamaServerAsync(string preferredFlavor = "cuda-12.4", bool allowAnyExisting = true)
    {
        var existing = FindLlamaServer(preferredFlavor) ?? (allowAnyExisting ? FindLlamaServer() : null);
        if (existing is not null)
        {
            return existing;
        }

        try
        {
            return await InstallLlamaCppAsync(preferredFlavor);
        }
        catch when (preferredFlavor != "cpu")
        {
            return await InstallLlamaCppAsync("cpu");
        }
    }

    private string? FindLlamaServer(string? flavor = null)
    {
        var llamaRoot = Path.Combine(runtimeDir, "llama.cpp");
        if (!Directory.Exists(llamaRoot))
        {
            return null;
        }

        var files = Directory.GetFiles(llamaRoot, "llama-server.exe", SearchOption.AllDirectories)
            .Where(path => flavor is null || path.Contains(flavor, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(File.GetLastWriteTimeUtc)
            .ToArray();
        return files.FirstOrDefault();
    }

    private async Task<string> InstallLlamaCppAsync(string flavor)
    {
        ShowStatus($"Downloading llama.cpp {flavor}...");
        using var client = CreateHttpClient();
        using var response = await client.GetAsync("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest");
        response.EnsureSuccessStatusCode();
        using var release = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        var tag = release.RootElement.GetProperty("tag_name").GetString() ?? "latest";
        var installDir = Path.Combine(runtimeDir, "llama.cpp", $"{tag}-{flavor}");
        Directory.CreateDirectory(installDir);

        foreach (var pattern in AssetPatterns(flavor))
        {
            var asset = FindAsset(release.RootElement.GetProperty("assets"), pattern);
            if (asset is null)
            {
                throw new InvalidOperationException($"No llama.cpp asset matched {pattern}.");
            }

            var name = asset.Value.GetProperty("name").GetString()!;
            var url = asset.Value.GetProperty("browser_download_url").GetString()!;
            var target = Path.Combine(downloadsDir, name);
            await DownloadFileAsync(client, url, target, $"Downloading {name}");
            ZipFile.ExtractToDirectory(target, installDir, overwriteFiles: true);
        }

        var server = Directory.GetFiles(installDir, "llama-server.exe", SearchOption.AllDirectories).FirstOrDefault();
        if (server is null)
        {
            throw new FileNotFoundException("llama-server.exe was not found after extraction.");
        }
        return server;
    }

    private static string[] AssetPatterns(string flavor) => flavor switch
    {
        "cuda-12.4" => new[]
        {
            "^llama-.+-bin-win-cuda-12\\.4-x64\\.zip$",
            "^cudart-llama-bin-win-cuda-12\\.4-x64\\.zip$",
        },
        "cpu" => new[] { "^llama-.+-bin-win-cpu-x64\\.zip$" },
        _ => new[] { "^llama-.+-bin-win-cpu-x64\\.zip$" },
    };

    private static JsonElement? FindAsset(JsonElement assets, string pattern)
    {
        var regex = new Regex(pattern, RegexOptions.IgnoreCase);
        foreach (var asset in assets.EnumerateArray())
        {
            var name = asset.GetProperty("name").GetString();
            if (name is not null && regex.IsMatch(name))
            {
                return asset;
            }
        }
        return null;
    }

    private async Task EnsureModelAsync(string modelPath)
    {
        if (File.Exists(modelPath) && new FileInfo(modelPath).Length > 0)
        {
            return;
        }

        Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);
        using var client = CreateHttpClient();
        await DownloadFileAsync(client, ModelUrl, modelPath, "Downloading Qwen3-8B model");
    }

    private async Task DownloadFileAsync(HttpClient client, string url, string target, string label)
    {
        if (File.Exists(target) && new FileInfo(target).Length > 0)
        {
            return;
        }

        var partial = target + ".partial";
        using var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();
        var total = response.Content.Headers.ContentLength;
        await using var input = await response.Content.ReadAsStreamAsync();
        await using var output = File.Create(partial);
        var buffer = new byte[1024 * 1024];
        long readTotal = 0;
        var lastNotice = DateTimeOffset.MinValue;
        while (true)
        {
            var read = await input.ReadAsync(buffer);
            if (read <= 0)
            {
                break;
            }
            await output.WriteAsync(buffer.AsMemory(0, read));
            readTotal += read;
            if (DateTimeOffset.Now - lastNotice > TimeSpan.FromSeconds(10))
            {
                lastNotice = DateTimeOffset.Now;
                var suffix = total.HasValue ? $" {readTotal / 1024 / 1024} MB / {total.Value / 1024 / 1024} MB" : $" {readTotal / 1024 / 1024} MB";
                ShowStatus(label + suffix);
            }
        }
        output.Close();
        if (File.Exists(target))
        {
            File.Delete(target);
        }
        File.Move(partial, target);
    }

    private async Task ConfigureModelAsync(string modelPath, string llamaServer)
    {
        var payload = new
        {
            enabled = true,
            provider = "openai_compatible",
            base_url = $"http://{Host}:{LlamaPort}/v1",
            model = modelPath,
            temperature = 0.2,
            timeout_seconds = 180,
            local_models_dir = modelsDir,
            local_model_path = modelPath,
            llama_server_path = llamaServer,
        };
        using var client = CreateHttpClient();
        var json = JsonSerializer.Serialize(payload);
        using var body = new StringContent(json, Encoding.UTF8, "application/json");
        await client.PatchAsync($"http://{Host}:{WebPort}/api/model/settings", body);
    }

    private static HttpClient CreateHttpClient()
    {
        var client = new HttpClient { Timeout = TimeSpan.FromHours(2) };
        client.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("LocalMathRAG", "0.1"));
        return client;
    }

    private static Process StartHidden(string fileName, string workingDirectory, string stdoutPath, string stderrPath, params string[] args)
    {
        var psi = new ProcessStartInfo
        {
            FileName = fileName,
            WorkingDirectory = workingDirectory,
            UseShellExecute = false,
            CreateNoWindow = true,
            WindowStyle = ProcessWindowStyle.Hidden,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (var arg in args)
        {
            psi.ArgumentList.Add(arg);
        }

        Directory.CreateDirectory(Path.GetDirectoryName(stdoutPath)!);
        var process = new Process { StartInfo = psi, EnableRaisingEvents = true };
        var stdout = new StreamWriter(new FileStream(stdoutPath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite), Encoding.UTF8) { AutoFlush = true };
        var stderr = new StreamWriter(new FileStream(stderrPath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite), Encoding.UTF8) { AutoFlush = true };
        process.OutputDataReceived += (_, e) => { if (e.Data is not null) stdout.WriteLine(e.Data); };
        process.ErrorDataReceived += (_, e) => { if (e.Data is not null) stderr.WriteLine(e.Data); };
        process.Exited += (_, _) =>
        {
            stdout.Dispose();
            stderr.Dispose();
        };
        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        return process;
    }

    private static async Task<bool> IsHttpReadyAsync(string url, TimeSpan timeout)
    {
        using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(3) };
        var deadline = DateTimeOffset.UtcNow + timeout;
        while (DateTimeOffset.UtcNow < deadline)
        {
            try
            {
                using var response = await client.GetAsync(url);
                if (response.IsSuccessStatusCode)
                {
                    return true;
                }
            }
            catch
            {
            }
            await Task.Delay(1000);
        }
        return false;
    }

    private void OpenWebApp() => OpenUrl($"http://{Host}:{WebPort}");

    private void OpenLogs()
    {
        var launcherLog = Path.Combine(dataDir, "launcher.err.log");
        if (File.Exists(launcherLog))
        {
            OpenPath(launcherLog);
            return;
        }
        OpenPath(dataDir);
    }

    private static void OpenUrl(string url)
    {
        Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
    }

    private static void OpenPath(string path)
    {
        Process.Start(new ProcessStartInfo(path) { UseShellExecute = true });
    }

    private void ShowStatus(string message)
    {
        try
        {
            notifyIcon.BalloonTipTitle = "LocalMathRAG";
            notifyIcon.BalloonTipText = message;
            notifyIcon.ShowBalloonTip(3000);
            File.AppendAllText(Path.Combine(dataDir, "launcher.out.log"), DateTimeOffset.Now + " " + message + Environment.NewLine, Encoding.UTF8);
        }
        catch
        {
        }
    }

    private static string ResolveRootDir()
    {
        var configured = Environment.GetEnvironmentVariable("LOCALMATHRAG_ROOT");
        if (!string.IsNullOrWhiteSpace(configured))
        {
            return Path.GetFullPath(configured);
        }
        return AppContext.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
    }
}
