using System.Diagnostics;
using System.Text;
using System.Windows.Forms;

namespace LocalMathRAGFlow;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
        ApplicationConfiguration.Initialize();
        Application.Run(new TrayContext());
    }
}

internal sealed class TrayContext : ApplicationContext
{
    private readonly NotifyIcon tray;
    private readonly ToolStripMenuItem statusItem;
    private readonly ToolStripMenuItem startItem;
    private readonly ToolStripMenuItem stopItem;
    private readonly ToolStripMenuItem restartItem;
    private readonly string root;
    private readonly string logDir;
    private CancellationTokenSource? currentRun;

    public TrayContext()
    {
        root = FindRoot();
        logDir = Path.Combine(root, "data", "launcher");
        Directory.CreateDirectory(logDir);

        statusItem = new ToolStripMenuItem("Status: starting") { Enabled = false };
        startItem = MenuItem("Start services", async () => await StartServicesAsync(openBrowser: true));
        stopItem = MenuItem("Stop services", async () => await StopServicesAsync());
        restartItem = MenuItem("Restart services", async () => await RestartServicesAsync());

        tray = new NotifyIcon
        {
            Icon = System.Drawing.SystemIcons.Application,
            Text = "LocalMathRAGFlow",
            Visible = true,
            ContextMenuStrip = new ContextMenuStrip()
        };
        tray.ContextMenuStrip.Items.Add(statusItem);
        tray.ContextMenuStrip.Items.Add(new ToolStripSeparator());
        tray.ContextMenuStrip.Items.Add(MenuItem("Open RAGFlow", () => OpenUrl(WebUrl)));
        tray.ContextMenuStrip.Items.Add(MenuItem("Open object service", () => OpenUrl("http://127.0.0.1:8088/health")));
        tray.ContextMenuStrip.Items.Add(startItem);
        tray.ContextMenuStrip.Items.Add(stopItem);
        tray.ContextMenuStrip.Items.Add(restartItem);
        tray.ContextMenuStrip.Items.Add(new ToolStripSeparator());
        tray.ContextMenuStrip.Items.Add(MenuItem("Open data directory", () => OpenPath(Path.Combine(root, "data"))));
        tray.ContextMenuStrip.Items.Add(MenuItem("View launcher log", () => OpenPath(LogFile)));
        tray.ContextMenuStrip.Items.Add(MenuItem("View compose log", () => OpenPath(ComposeLogFile)));
        tray.ContextMenuStrip.Items.Add(new ToolStripSeparator());
        tray.ContextMenuStrip.Items.Add(MenuItem("Exit", async () => await ExitAsync()));
        tray.DoubleClick += (_, _) => OpenUrl(WebUrl);

        _ = StartServicesAsync(openBrowser: true);
    }

    private static ToolStripMenuItem MenuItem(string text, Action action)
    {
        var item = new ToolStripMenuItem(text);
        item.Click += (_, _) => action();
        return item;
    }

    private static ToolStripMenuItem MenuItem(string text, Func<Task> action)
    {
        var item = new ToolStripMenuItem(text);
        item.Click += async (_, _) => await action();
        return item;
    }

    private string WebUrl => $"http://127.0.0.1:{GetRagflowWebPort()}";
    private string LogFile => Path.Combine(logDir, "launcher.log");
    private string ComposeLogFile => Path.Combine(logDir, "compose.log");

    private async Task StartServicesAsync(bool openBrowser)
    {
        if (currentRun is not null)
        {
            SetStatus("already running a task");
            return;
        }

        currentRun = new CancellationTokenSource();
        SetBusy(true);
        try
        {
            SetStatus("checking Docker");
            if (!CommandExists("docker"))
            {
                MessageBox.Show(
                    "Docker CLI was not found. Please install Docker Desktop first.",
                    "LocalMathRAGFlow",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Warning);
                return;
            }

            if (!await DockerReadyAsync(currentRun.Token))
            {
                SetStatus("starting Docker Desktop");
                StartDockerDesktop();
                if (!await WaitForDockerAsync(TimeSpan.FromMinutes(5), currentRun.Token))
                {
                    MessageBox.Show(
                        "Docker Desktop did not become ready within 5 minutes. Please check Docker Desktop and try again.",
                        "LocalMathRAGFlow",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Warning);
                    return;
                }
            }

            if (!Directory.Exists(RagflowDockerDir))
            {
                var result = MessageBox.Show(
                    "RAGFlow source is missing. Download it now from GitHub?",
                    "LocalMathRAGFlow",
                    MessageBoxButtons.YesNo,
                    MessageBoxIcon.Question);
                if (result != DialogResult.Yes)
                {
                    return;
                }
                SetStatus("downloading RAGFlow");
                await RunPowerShellScriptAsync(Path.Combine(root, "scripts", "bootstrap-ragflow.ps1"), "", currentRun.Token);
            }

            SetStatus("starting containers");
            await RunDockerComposeAsync("up -d", currentRun.Token);
            SetStatus("running");
            tray.ShowBalloonTip(3000, "LocalMathRAGFlow", "RAGFlow services are starting. The first run may take a while.", ToolTipIcon.Info);
            if (openBrowser)
            {
                await Task.Delay(1500, currentRun.Token);
                OpenUrl(WebUrl);
            }
        }
        catch (OperationCanceledException)
        {
            SetStatus("cancelled");
        }
        catch (Exception ex)
        {
            Log("ERROR " + ex);
            SetStatus("error");
            MessageBox.Show(ex.Message, "LocalMathRAGFlow", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            SetBusy(false);
            currentRun?.Dispose();
            currentRun = null;
        }
    }

    private async Task StopServicesAsync()
    {
        SetBusy(true);
        try
        {
            currentRun?.Cancel();
            SetStatus("stopping containers");
            await RunDockerComposeAsync("down", CancellationToken.None);
            SetStatus("stopped");
        }
        catch (Exception ex)
        {
            Log("ERROR " + ex);
            MessageBox.Show(ex.Message, "LocalMathRAGFlow", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            SetBusy(false);
        }
    }

    private async Task RestartServicesAsync()
    {
        await StopServicesAsync();
        await StartServicesAsync(openBrowser: true);
    }

    private async Task ExitAsync()
    {
        var result = MessageBox.Show(
            "Stop Docker services before exit?",
            "LocalMathRAGFlow",
            MessageBoxButtons.YesNoCancel,
            MessageBoxIcon.Question);
        if (result == DialogResult.Cancel)
        {
            return;
        }
        if (result == DialogResult.Yes)
        {
            await StopServicesAsync();
        }
        tray.Visible = false;
        tray.Dispose();
        ExitThread();
    }

    private string RagflowDockerDir => Path.Combine(root, "third_party", "ragflow", "docker");
    private string OverrideCompose => Path.Combine(root, "docker", "docker-compose.localmathrag.yml");

    private async Task RunDockerComposeAsync(string args, CancellationToken token)
    {
        var composeArgs = $"compose -f docker-compose.yml -f \"{OverrideCompose}\" {args}";
        var env = new Dictionary<string, string?>
        {
            ["LOCALMATHRAG_ROOT"] = root,
            ["DOC_ENGINE"] = "elasticsearch",
            ["DEVICE"] = "cpu",
            ["COMPOSE_PROFILES"] = "elasticsearch,cpu"
        };
        await RunProcessAsync("docker", composeArgs, RagflowDockerDir, ComposeLogFile, env, token);
    }

    private async Task RunPowerShellScriptAsync(string script, string args, CancellationToken token)
    {
        var psArgs = $"-NoProfile -ExecutionPolicy Bypass -File \"{script}\" {args}";
        await RunProcessAsync("powershell", psArgs, root, LogFile, null, token);
    }

    private async Task<bool> DockerReadyAsync(CancellationToken token)
    {
        try
        {
            var code = await RunProcessAsync("docker", "info", root, LogFile, null, token, throwOnError: false);
            return code == 0;
        }
        catch
        {
            return false;
        }
    }

    private async Task<bool> WaitForDockerAsync(TimeSpan timeout, CancellationToken token)
    {
        var start = DateTimeOffset.UtcNow;
        while (DateTimeOffset.UtcNow - start < timeout)
        {
            if (await DockerReadyAsync(token))
            {
                return true;
            }
            await Task.Delay(3000, token);
        }
        return false;
    }

    private static void StartDockerDesktop()
    {
        var candidates = new[]
        {
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "Docker", "Docker", "Docker Desktop.exe"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Docker", "Docker Desktop.exe")
        };
        var dockerDesktop = candidates.FirstOrDefault(File.Exists);
        if (dockerDesktop is null)
        {
            throw new FileNotFoundException("Docker Desktop was not found. Please install Docker Desktop first.");
        }
        Process.Start(new ProcessStartInfo
        {
            FileName = dockerDesktop,
            UseShellExecute = true,
            WindowStyle = ProcessWindowStyle.Minimized
        });
    }

    private async Task<int> RunProcessAsync(
        string fileName,
        string arguments,
        string workingDirectory,
        string logFile,
        IReadOnlyDictionary<string, string?>? environment,
        CancellationToken token,
        bool throwOnError = true)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(logFile)!);
        await File.AppendAllTextAsync(logFile, $"\n[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {fileName} {arguments}\n", Encoding.UTF8, token);

        var psi = new ProcessStartInfo
        {
            FileName = fileName,
            Arguments = arguments,
            WorkingDirectory = workingDirectory,
            CreateNoWindow = true,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8
        };
        if (environment is not null)
        {
            foreach (var kv in environment)
            {
                psi.Environment[kv.Key] = kv.Value;
            }
        }

        using var process = new Process { StartInfo = psi, EnableRaisingEvents = true };
        process.OutputDataReceived += (_, e) => { if (e.Data is not null) LogTo(logFile, e.Data); };
        process.ErrorDataReceived += (_, e) => { if (e.Data is not null) LogTo(logFile, e.Data); };

        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        await process.WaitForExitAsync(token);

        if (throwOnError && process.ExitCode != 0)
        {
            throw new InvalidOperationException($"{fileName} exited with code {process.ExitCode}. See {logFile}");
        }
        return process.ExitCode;
    }

    private void SetBusy(bool busy)
    {
        startItem.Enabled = !busy;
        stopItem.Enabled = !busy;
        restartItem.Enabled = !busy;
    }

    private void SetStatus(string status)
    {
        statusItem.Text = "Status: " + status;
        tray.Text = status.Length > 40 ? "LocalMathRAGFlow" : $"LocalMathRAGFlow - {status}";
        Log("STATUS " + status);
    }

    private void Log(string message) => LogTo(LogFile, message);

    private static void LogTo(string file, string message)
    {
        try
        {
            File.AppendAllText(file, $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {message}\n", Encoding.UTF8);
        }
        catch
        {
            // Logging should never break the tray app.
        }
    }

    private static bool CommandExists(string command)
    {
        try
        {
            using var process = Process.Start(new ProcessStartInfo
            {
                FileName = "where",
                Arguments = command,
                CreateNoWindow = true,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            });
            process?.WaitForExit(3000);
            return process?.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }

    private int GetRagflowWebPort()
    {
        var envPath = Path.Combine(RagflowDockerDir, ".env");
        if (!File.Exists(envPath))
        {
            return 80;
        }
        foreach (var line in File.ReadLines(envPath))
        {
            if (line.StartsWith("SVR_WEB_HTTP_PORT=", StringComparison.Ordinal))
            {
                var raw = line.Split('=', 2)[1].Trim();
                if (int.TryParse(raw, out var port))
                {
                    return port;
                }
            }
        }
        return 80;
    }

    private static string FindRoot()
    {
        var current = AppContext.BaseDirectory;
        for (var dir = new DirectoryInfo(current); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "docker", "docker-compose.localmathrag.yml")) &&
                Directory.Exists(Path.Combine(dir.FullName, "scripts")))
            {
                return dir.FullName;
            }
        }
        return Directory.GetCurrentDirectory();
    }

    private static void OpenUrl(string url) => Process.Start(new ProcessStartInfo { FileName = url, UseShellExecute = true });

    private static void OpenPath(string path)
    {
        if (!File.Exists(path) && !Directory.Exists(path))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path) ?? path);
            File.WriteAllText(path, "", Encoding.UTF8);
        }
        Process.Start(new ProcessStartInfo { FileName = path, UseShellExecute = true });
    }
}
