using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Net.Http.Json;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
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

internal sealed partial class TrayContext : ApplicationContext
{
    private readonly NotifyIcon tray;
    private readonly ContextMenuStrip menu;
    private readonly Form menuHost;
    private readonly ToolStripMenuItem statusItem;
    private readonly ToolStripMenuItem startItem;
    private readonly ToolStripMenuItem stopItem;
    private readonly ToolStripMenuItem restartItem;
    private readonly SynchronizationContext uiContext;
    private readonly string root;
    private readonly string logDir;
    private readonly Icon appIcon;
    private Process? browserProcess;
    private string lastStatus = "";
    private CancellationTokenSource? currentRun;

    public TrayContext()
    {
        uiContext = SynchronizationContext.Current ?? new WindowsFormsSynchronizationContext();
        root = FindRoot();
        logDir = Path.Combine(root, "data", "launcher");
        Directory.CreateDirectory(logDir);
        appIcon = LoadAppIcon(root);

        statusItem = new ToolStripMenuItem("Status: starting")
        {
            Enabled = false,
            Font = new Font("Segoe UI Semibold", 10F, FontStyle.Bold),
            Image = MenuGlyph.Status(Color.FromArgb(22, 119, 255)),
            Padding = new Padding(4, 8, 8, 8)
        };
        startItem = MenuItem("Start services", async () => await StartServicesAsync(openBrowser: true), MenuGlyph.Play());
        stopItem = MenuItem("Stop services", async () => await StopServicesAsync(), MenuGlyph.Stop());
        restartItem = MenuItem("Restart services", async () => await RestartServicesAsync(), MenuGlyph.Restart());
        menuHost = CreateMenuHost();
        menu = CreateMenu();

        tray = new NotifyIcon
        {
            Icon = appIcon,
            Text = "LocalMathRAGFlow",
            ContextMenuStrip = menu,
            Visible = true
        };
        tray.MouseClick += (_, e) =>
        {
            if (e.Button == MouseButtons.Right)
            {
                ShowTrayMenu();
            }
        };
        tray.MouseUp += (_, e) =>
        {
            if (e.Button == MouseButtons.Right)
            {
                ShowTrayMenu();
            }
        };
        menu.Items.Add(statusItem);
        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add(MenuItem("Open RAGFlow", async () => await OpenRagflowAsync(CancellationToken.None), MenuGlyph.Open()));
        menu.Items.Add(MenuItem("Open object service", () => OpenUrl("http://127.0.0.1:8088/health"), MenuGlyph.Link()));
        menu.Items.Add(startItem);
        menu.Items.Add(stopItem);
        menu.Items.Add(restartItem);
        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add(MenuItem("Open data directory", () => OpenPath(Path.Combine(root, "data")), MenuGlyph.Folder()));
        menu.Items.Add(MenuItem("View launcher log", () => OpenPath(LogFile), MenuGlyph.Log()));
        menu.Items.Add(MenuItem("View compose log", () => OpenPath(ComposeLogFile), MenuGlyph.Log()));
        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add(MenuItem("Exit", async () => await ExitAsync(), MenuGlyph.Exit()));
        tray.MouseDoubleClick += async (_, e) =>
        {
            if (e.Button == MouseButtons.Left)
            {
                await OpenRagflowAsync(CancellationToken.None);
            }
        };

        _ = StartServicesAsync(openBrowser: true);
    }

    private static ContextMenuStrip CreateMenu()
    {
        return new ContextMenuStrip
        {
            BackColor = Color.FromArgb(250, 252, 255),
            ForeColor = Color.FromArgb(16, 24, 40),
            Font = new Font("Segoe UI", 10F, FontStyle.Regular),
            Padding = new Padding(8, 8, 8, 8),
            ShowImageMargin = true,
            Renderer = new ModernMenuRenderer()
        };
    }

    private static ToolStripMenuItem MenuItem(string text, Action action, Image? image = null)
    {
        var item = new ToolStripMenuItem(text) { Image = image, Padding = new Padding(4, 6, 8, 6) };
        item.Click += (_, _) => action();
        return item;
    }

    private static ToolStripMenuItem MenuItem(string text, Func<Task> action, Image? image = null)
    {
        var item = new ToolStripMenuItem(text) { Image = image, Padding = new Padding(4, 6, 8, 6) };
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

            var allowLlama = await ConfirmLocalModelRuntimeAsync(currentRun.Token);
            SetStatus("starting containers");
            await RunDockerComposeAsync("up -d --build", currentRun.Token, allowLlama);
            SetStatus("waiting for RAGFlow web");
            var webReady = await WaitForHttpOkAsync($"{WebUrl}/api/v1/system/config", TimeSpan.FromMinutes(3), currentRun.Token);
            SetStatus("running");
            if (!webReady)
            {
                tray.ShowBalloonTip(3000, "LocalMathRAGFlow", "Containers are running, but RAGFlow Web may still be warming up.", ToolTipIcon.Warning);
            }
            if (openBrowser)
            {
                await OpenRagflowAsync(currentRun.Token);
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
        appIcon.Dispose();
        menu.Dispose();
        tray.Dispose();
        menuHost.Dispose();
        ExitThread();
    }

    private string RagflowDockerDir => Path.Combine(root, "third_party", "ragflow", "docker");
    private string OverrideCompose => Path.Combine(root, "docker", "docker-compose.localmathrag.yml");
    private string WebDistCompose => Path.Combine(root, "docker", "docker-compose.webdist.yml");

    private static Icon LoadAppIcon(string root)
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "Assets", "ragflow.ico"),
            Path.Combine(root, "launcher", "LocalMathRAGFlow", "Assets", "ragflow.ico"),
            Path.Combine(root, "Assets", "ragflow.ico")
        };
        var iconPath = candidates.FirstOrDefault(File.Exists);
        return iconPath is null ? SystemIcons.Application : new Icon(iconPath);
    }

    private async Task OpenRagflowAsync(CancellationToken token)
    {
        var loginUrl = await TryBuildAutoLoginUrlAsync(token);
        OpenOrFocusAppWindow(loginUrl ?? WebUrl);
    }

    private void ShowTrayMenu()
    {
        uiContext.Post(_ =>
        {
            if (menu.Visible)
            {
                menu.Close();
            }
            var position = Cursor.Position;
            menuHost.Location = new Point(position.X, position.Y);
            SetForegroundWindow(menuHost.Handle);
            menu.Show(menuHost, menuHost.PointToClient(position));
        }, null);
    }

    private async Task<string?> TryBuildAutoLoginUrlAsync(CancellationToken token)
    {
        try
        {
            using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
            var payload = new
            {
                email = "admin@ragflow.io",
                password = EncryptRagflowPassword("admin")
            };
            using var response = await client.PostAsJsonAsync($"{WebUrl}/api/v1/auth/login", payload, token);
            var content = await response.Content.ReadAsStringAsync(token);
            if (!response.IsSuccessStatusCode || !content.Contains("\"code\":0", StringComparison.Ordinal))
            {
                Log("AUTOLOGIN failed: " + content);
                return null;
            }
            if (!response.Headers.TryGetValues("Authorization", out var values))
            {
                Log("AUTOLOGIN failed: missing Authorization header");
                return null;
            }
            var auth = values.FirstOrDefault();
            return string.IsNullOrWhiteSpace(auth) ? null : $"{WebUrl}/?auth={Uri.EscapeDataString(auth)}";
        }
        catch (Exception ex)
        {
            Log("AUTOLOGIN failed: " + ex.Message);
            return null;
        }
    }

    private static string EncryptRagflowPassword(string password)
    {
        const string publicKey = """
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEArq9XTUSeYr2+N1h3Afl/z8Dse/2yD0ZGrKwx+EEEcdsBLca9Ynmx3nIB5obmLlSfmskLpBo0UACBmB5rEjBp2Q2f3AG3Hjd4B+gNCG6BDaawuDlgANIhGnaTLrIqWrrcm4EMzJOnAOI1fgzJRsOOUEfaS318Eq9OVO3apEyCCt0lOQK6PuksduOjVxtltDav+guVAA068NrPYmRNabVKRNLJpL8w4D44sfth5RvZ3q9t+6RTArpEtc5sh5ChzvqPOzKGMXW83C95TxmXqpbK6olN4RevSfVjEAgCydH6HN6OhtOQEcnrU97r9H0iZOWwbw3pVrZiUkuRD1R56Wzs2wIDAQAB
-----END PUBLIC KEY-----
""";
        using var rsa = RSA.Create();
        rsa.ImportFromPem(publicKey);
        var base64Password = Convert.ToBase64String(Encoding.UTF8.GetBytes(password));
        var encrypted = rsa.Encrypt(Encoding.UTF8.GetBytes(base64Password), RSAEncryptionPadding.Pkcs1);
        return Convert.ToBase64String(encrypted);
    }

    private async Task RunDockerComposeAsync(string args, CancellationToken token, bool allowLlama = true)
    {
        var composeFiles = $"-f docker-compose.yml -f \"{OverrideCompose}\"";
        if (Directory.Exists(Path.Combine(root, "third_party", "ragflow", "web", "dist")) &&
            File.Exists(WebDistCompose))
        {
            composeFiles += $" -f \"{WebDistCompose}\"";
        }
        var composeArgs = $"compose {composeFiles} {args}";
        var profiles = BuildComposeProfiles(allowLlama);
        var env = new Dictionary<string, string?>
        {
            ["LOCALMATHRAG_ROOT"] = root,
            ["DOC_ENGINE"] = "elasticsearch",
            ["DEVICE"] = "cpu",
            ["COMPOSE_PROFILES"] = string.Join(",", profiles)
        };
        var modelName = FindDefaultGgufModelName();
        if (modelName is not null)
        {
            env["LOCALMATHRAG_GGUF_MODEL"] = modelName;
        }
        Action<string>? onOutput = args.StartsWith("up", StringComparison.OrdinalIgnoreCase) ? HandleComposeOutput : null;
        await RunProcessAsync("docker", composeArgs, RagflowDockerDir, ComposeLogFile, env, token, onOutput);
    }

    private List<string> BuildComposeProfiles(bool allowLlama)
    {
        var profiles = new List<string> { "elasticsearch", "cpu" };
        var modelName = FindDefaultGgufModelName();
        if (!allowLlama || modelName is null)
        {
            return profiles;
        }

        var llamaProfile = Environment.GetEnvironmentVariable("LOCALMATHRAG_LLAMA_PROFILE");
        if (string.Equals(llamaProfile, "none", StringComparison.OrdinalIgnoreCase))
        {
            return profiles;
        }
        profiles.Add(string.Equals(llamaProfile, "cuda", StringComparison.OrdinalIgnoreCase)
            ? "llama-cpp-cuda"
            : "llama-cpp-cpu");
        return profiles;
    }

    private async Task<bool> ConfirmLocalModelRuntimeAsync(CancellationToken token)
    {
        var modelName = FindDefaultGgufModelName();
        if (modelName is null)
        {
            return false;
        }
        if (string.Equals(Environment.GetEnvironmentVariable("LOCALMATHRAG_LLAMA_PROFILE"), "none", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        var image = GetLlamaDockerImage();
        if (await DockerImageExistsAsync(image, token))
        {
            return true;
        }

        var result = MessageBox.Show(
            $"Detected local model {modelName}, but the local llama.cpp Docker image is not installed yet.\n\nDownload image now?\n\nChoose No to start RAGFlow without the local model endpoint.",
            "LocalMathRAGFlow",
            MessageBoxButtons.YesNo,
            MessageBoxIcon.Question);
        return result == DialogResult.Yes;
    }

    private string GetLlamaDockerImage()
    {
        return string.Equals(Environment.GetEnvironmentVariable("LOCALMATHRAG_LLAMA_PROFILE"), "cuda", StringComparison.OrdinalIgnoreCase)
            ? "ghcr.io/ggml-org/llama.cpp:server-cuda"
            : "ghcr.io/ggml-org/llama.cpp:server";
    }

    private async Task<bool> DockerImageExistsAsync(string image, CancellationToken token)
    {
        var code = await RunProcessAsync("docker", $"image inspect \"{image}\"", root, LogFile, null, token, throwOnError: false);
        return code == 0;
    }

    private string? FindDefaultGgufModelName()
    {
        var modelDir = Path.Combine(root, "data", "models");
        if (!Directory.Exists(modelDir))
        {
            return null;
        }
        return Directory.EnumerateFiles(modelDir, "*.gguf", SearchOption.TopDirectoryOnly)
            .Select(Path.GetFileName)
            .FirstOrDefault(name => !string.IsNullOrWhiteSpace(name));
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

    private static async Task<bool> WaitForHttpOkAsync(string url, TimeSpan timeout, CancellationToken token)
    {
        using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
        var start = DateTimeOffset.UtcNow;
        while (DateTimeOffset.UtcNow - start < timeout)
        {
            try
            {
                using var response = await client.GetAsync(url, token);
                if ((int)response.StatusCode < 500)
                {
                    return true;
                }
            }
            catch
            {
                // The web container can be up before the app is ready to accept requests.
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
        => await RunProcessAsync(fileName, arguments, workingDirectory, logFile, environment, token, null, throwOnError);

    private async Task<int> RunProcessAsync(
        string fileName,
        string arguments,
        string workingDirectory,
        string logFile,
        IReadOnlyDictionary<string, string?>? environment,
        CancellationToken token,
        Action<string>? onOutput,
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
        process.OutputDataReceived += (_, e) =>
        {
            if (e.Data is not null)
            {
                LogTo(logFile, e.Data);
                onOutput?.Invoke(e.Data);
            }
        };
        process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data is not null)
            {
                LogTo(logFile, e.Data);
                onOutput?.Invoke(e.Data);
            }
        };

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

    private void HandleComposeOutput(string line)
    {
        if (line.Contains("Downloading", StringComparison.OrdinalIgnoreCase) ||
            line.Contains("Pulling", StringComparison.OrdinalIgnoreCase))
        {
            SetStatus("downloading Docker images");
        }
        else if (line.Contains("Extracting", StringComparison.OrdinalIgnoreCase))
        {
            SetStatus("extracting Docker images");
        }
        else if (line.Contains("Building", StringComparison.OrdinalIgnoreCase) ||
                 line.Contains("=>", StringComparison.Ordinal))
        {
            SetStatus("building local services");
        }
        else if (line.Contains("Creating", StringComparison.OrdinalIgnoreCase) ||
                 line.Contains("Recreating", StringComparison.OrdinalIgnoreCase))
        {
            SetStatus("creating containers");
        }
        else if (line.Contains("Starting", StringComparison.OrdinalIgnoreCase))
        {
            SetStatus("starting containers");
        }
        else if (line.Contains("Started", StringComparison.OrdinalIgnoreCase) ||
                 line.Contains("Running", StringComparison.OrdinalIgnoreCase))
        {
            SetStatus("containers running");
        }
    }

    private void SetBusy(bool busy)
    {
        startItem.Enabled = !busy;
        stopItem.Enabled = !busy;
        restartItem.Enabled = !busy;
    }

    private void SetStatus(string status)
    {
        if (SynchronizationContext.Current != uiContext)
        {
            uiContext.Post(_ => SetStatus(status), null);
            return;
        }
        if (status == lastStatus)
        {
            return;
        }
        lastStatus = status;
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
        var candidates = new List<string>();
        for (var dir = new DirectoryInfo(current); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "docker", "docker-compose.localmathrag.yml")) &&
                Directory.Exists(Path.Combine(dir.FullName, "scripts")))
            {
                candidates.Add(dir.FullName);
            }
        }

        var installedRoot = candidates.FirstOrDefault(candidate =>
            File.Exists(Path.Combine(candidate, "third_party", "ragflow", "docker", "docker-compose.yml")));
        if (installedRoot is not null)
        {
            return installedRoot;
        }

        return candidates.FirstOrDefault() ?? Directory.GetCurrentDirectory();
    }

    private static Form CreateMenuHost()
    {
        var form = new Form
        {
            FormBorderStyle = FormBorderStyle.FixedToolWindow,
            ShowInTaskbar = false,
            StartPosition = FormStartPosition.Manual,
            Size = new Size(1, 1),
            Location = new Point(-32000, -32000),
            Opacity = 0
        };
        form.Show();
        form.Hide();
        return form;
    }

    private void OpenOrFocusAppWindow(string url)
    {
        if (TryFocusBrowserProcess())
        {
            return;
        }

        var browser = FindChromiumBrowser();
        if (browser is not null)
        {
            var profileDir = Path.Combine(root, "data", "launcher", "browser-profile");
            Directory.CreateDirectory(profileDir);
            browserProcess = Process.Start(new ProcessStartInfo
            {
                FileName = browser,
                Arguments = $"--app=\"{url}\" --user-data-dir=\"{profileDir}\" --no-first-run",
                UseShellExecute = false
            });
            return;
        }

        OpenUrl(url);
    }

    private bool TryFocusBrowserProcess()
    {
        try
        {
            if (browserProcess is null || browserProcess.HasExited)
            {
                return false;
            }
            browserProcess.Refresh();
            if (browserProcess.MainWindowHandle == IntPtr.Zero)
            {
                return false;
            }
            ShowWindow(browserProcess.MainWindowHandle, ShowWindowRestore);
            SetForegroundWindow(browserProcess.MainWindowHandle);
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static string? FindChromiumBrowser()
    {
        var candidates = new[]
        {
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), "Microsoft", "Edge", "Application", "msedge.exe"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "Microsoft", "Edge", "Application", "msedge.exe"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Microsoft", "Edge", "Application", "msedge.exe"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "Google", "Chrome", "Application", "chrome.exe"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), "Google", "Chrome", "Application", "chrome.exe"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Google", "Chrome", "Application", "chrome.exe")
        };
        return candidates.FirstOrDefault(File.Exists);
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

internal sealed partial class TrayContext
{
    private const int ShowWindowRestore = 9;

    [DllImport("user32.dll")]
    private static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll")]
    private static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
}

internal sealed class ModernMenuRenderer : ToolStripProfessionalRenderer
{
    private static readonly Color HoverBack = Color.FromArgb(232, 243, 255);
    private static readonly Color PressedBack = Color.FromArgb(213, 231, 255);
    private static readonly Color Border = Color.FromArgb(214, 224, 240);
    private static readonly Color ImageMargin = Color.FromArgb(245, 248, 252);

    public ModernMenuRenderer() : base(new ModernColorTable())
    {
        RoundedEdges = true;
    }

    protected override void OnRenderToolStripBorder(ToolStripRenderEventArgs e)
    {
        using var pen = new Pen(Border);
        var rect = new Rectangle(0, 0, e.ToolStrip.Width - 1, e.ToolStrip.Height - 1);
        e.Graphics.DrawRectangle(pen, rect);
    }

    protected override void OnRenderImageMargin(ToolStripRenderEventArgs e)
    {
        using var brush = new SolidBrush(ImageMargin);
        e.Graphics.FillRectangle(brush, e.AffectedBounds);
    }

    protected override void OnRenderMenuItemBackground(ToolStripItemRenderEventArgs e)
    {
        if (e.Item is not ToolStripMenuItem item || !item.Selected)
        {
            base.OnRenderMenuItemBackground(e);
            return;
        }

        var bounds = new Rectangle(6, 3, e.Item.Width - 12, e.Item.Height - 6);
        using var path = RoundedRect(bounds, 7);
        using var brush = new SolidBrush(item.Pressed ? PressedBack : HoverBack);
        e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
        e.Graphics.FillPath(brush, path);
    }

    protected override void OnRenderSeparator(ToolStripSeparatorRenderEventArgs e)
    {
        using var pen = new Pen(Border);
        var y = e.Item.Height / 2;
        e.Graphics.DrawLine(pen, 40, y, e.Item.Width - 10, y);
    }

    private static GraphicsPath RoundedRect(Rectangle rect, int radius)
    {
        var path = new GraphicsPath();
        var diameter = radius * 2;
        path.AddArc(rect.Left, rect.Top, diameter, diameter, 180, 90);
        path.AddArc(rect.Right - diameter, rect.Top, diameter, diameter, 270, 90);
        path.AddArc(rect.Right - diameter, rect.Bottom - diameter, diameter, diameter, 0, 90);
        path.AddArc(rect.Left, rect.Bottom - diameter, diameter, diameter, 90, 90);
        path.CloseFigure();
        return path;
    }

    private sealed class ModernColorTable : ProfessionalColorTable
    {
        public override Color MenuBorder => Border;
        public override Color ToolStripDropDownBackground => Color.FromArgb(250, 252, 255);
        public override Color ImageMarginGradientBegin => ImageMargin;
        public override Color ImageMarginGradientMiddle => ImageMargin;
        public override Color ImageMarginGradientEnd => ImageMargin;
        public override Color MenuItemSelected => HoverBack;
        public override Color MenuItemBorder => HoverBack;
        public override Color MenuItemPressedGradientBegin => PressedBack;
        public override Color MenuItemPressedGradientMiddle => PressedBack;
        public override Color MenuItemPressedGradientEnd => PressedBack;
    }
}

internal static class MenuGlyph
{
    private static readonly Color Blue = Color.FromArgb(22, 119, 255);
    private static readonly Color Dark = Color.FromArgb(52, 64, 84);
    private static readonly Color Red = Color.FromArgb(217, 45, 32);

    public static Image Open() => Draw(g =>
    {
        using var pen = Pen(Blue, 2.2f);
        g.DrawRectangle(pen, 4, 5, 12, 10);
        g.DrawLine(pen, 9, 10, 16, 3);
        g.DrawLine(pen, 12, 3, 16, 3);
        g.DrawLine(pen, 16, 3, 16, 7);
    });

    public static Image Link() => Draw(g =>
    {
        using var pen = Pen(Blue, 2.2f);
        g.DrawArc(pen, 3, 5, 9, 9, 120, 220);
        g.DrawArc(pen, 8, 5, 9, 9, -60, 220);
        g.DrawLine(pen, 8, 10, 12, 10);
    });

    public static Image Folder() => Draw(g =>
    {
        using var pen = Pen(Blue, 2f);
        using var brush = new SolidBrush(Color.FromArgb(232, 243, 255));
        var body = new Rectangle(3, 7, 14, 9);
        g.FillRectangle(brush, body);
        g.DrawRectangle(pen, body);
        g.DrawLine(pen, 4, 7, 7, 4);
        g.DrawLine(pen, 7, 4, 11, 4);
        g.DrawLine(pen, 11, 4, 13, 7);
    });

    public static Image Log() => Draw(g =>
    {
        using var pen = Pen(Dark, 1.8f);
        g.DrawRectangle(pen, 5, 3, 10, 14);
        g.DrawLine(pen, 7, 7, 13, 7);
        g.DrawLine(pen, 7, 10, 13, 10);
        g.DrawLine(pen, 7, 13, 11, 13);
    });

    public static Image Exit() => Draw(g =>
    {
        using var pen = Pen(Red, 2.2f);
        g.DrawLine(pen, 6, 6, 14, 14);
        g.DrawLine(pen, 14, 6, 6, 14);
    });

    public static Image Play() => Draw(g =>
    {
        using var brush = new SolidBrush(Blue);
        var points = new[] { new PointF(7, 5), new PointF(15, 10), new PointF(7, 15) };
        g.FillPolygon(brush, points);
    });

    public static Image Stop() => Draw(g =>
    {
        using var brush = new SolidBrush(Dark);
        g.FillRectangle(brush, 6, 6, 9, 9);
    });

    public static Image Restart() => Draw(g =>
    {
        using var pen = Pen(Blue, 2.1f);
        g.DrawArc(pen, 4, 4, 12, 12, 35, 285);
        using var brush = new SolidBrush(Blue);
        var points = new[] { new PointF(13, 3), new PointF(17, 4), new PointF(15, 8) };
        g.FillPolygon(brush, points);
    });

    public static Image Status(Color color) => Draw(g =>
    {
        using var brush = new SolidBrush(color);
        g.FillEllipse(brush, 6, 6, 8, 8);
    });

    private static Bitmap Draw(Action<Graphics> paint)
    {
        var bmp = new Bitmap(20, 20);
        using var g = Graphics.FromImage(bmp);
        g.SmoothingMode = SmoothingMode.AntiAlias;
        g.Clear(Color.Transparent);
        paint(g);
        return bmp;
    }

    private static Pen Pen(Color color, float width)
    {
        return new Pen(color, width)
        {
            StartCap = LineCap.Round,
            EndCap = LineCap.Round,
            LineJoin = LineJoin.Round
        };
    }
}
