using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Net;
using System.Net.Http.Json;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Windows.Forms;

namespace LocalMathRAGFlow;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
        ApplicationConfiguration.Initialize();
        Application.ThreadException += (_, e) => TrayContext.LogStartup("THREAD ERROR " + e.Exception);
        AppDomain.CurrentDomain.UnhandledException += (_, e) => TrayContext.LogStartup("UNHANDLED ERROR " + e.ExceptionObject);
        Application.ApplicationExit += (_, _) => TrayContext.LogStartup("APPLICATION EXIT");
        StartupStatusServer.StartDetached();
        BackgroundStartupTask.StartDetached();
        Application.Run(new TrayContext());
    }
}
internal sealed class TrayContext : ApplicationContext
{
    private NotifyIcon? tray;
    private readonly string logFile;
    private readonly System.Windows.Forms.Timer windowStateTimer;
    private Process? browserProcess;
    private DateTimeOffset lastOpenAt = DateTimeOffset.MinValue;

    public TrayContext()
    {
        logFile = Path.Combine(AppContext.BaseDirectory, "launcher-tray.log");
        Log("launcher tray starting");
        windowStateTimer = new System.Windows.Forms.Timer { Interval = 3000 };
        windowStateTimer.Tick += (_, _) => SaveCurrentBrowserWindowState();
        windowStateTimer.Start();

        var menu = CreateTrayMenu();
        menu.Opening += (_, _) => Log("menu opening");
        menu.Opened += (_, _) =>
        {
            ApplyRoundedRegion(menu);
            Log("menu opened");
        };
        AddHeader(menu, "LocalMathRAGFlow");
        AddTrayItem(menu, "Open RAGFlow", (_, _) => OpenWeb());
        AddTrayItem(menu, "Start services", (_, _) =>
        {
            BackgroundStartupTask.StartDetached();
            OpenWeb();
        });
        AddTrayItem(menu, "Stop services", async (_, _) =>
        {
            OpenWeb(forceStatusPage: true);
            await BackgroundStartupTask.StopServicesAsync();
        });
        AddSeparator(menu);
        AddTrayItem(menu, "Open dataset", (_, _) => OpenPath(Path.Combine(FindRoot(), "data", "dataset")));
        AddTrayItem(menu, "Open data directory", (_, _) => OpenPath(Path.Combine(FindRoot(), "data")));
        AddSeparator(menu);
        AddTrayItem(menu, "Exit", async (_, _) => await ExitAsync());

        tray = new NotifyIcon
        {
            Icon = LoadTrayIcon(),
            Text = "LocalMathRAGFlow",
            ContextMenuStrip = menu,
            Visible = true
        };
        tray.MouseClick += (_, e) =>
        {
            Log($"mouse click {e.Button} clicks={e.Clicks}");
            if (e.Button == MouseButtons.Left)
            {
                OpenWeb();
            }
        };
        tray.DoubleClick += (_, _) => OpenWeb();
        Log("launcher tray icon created");
        OpenWeb();
    }

    private static ContextMenuStrip CreateTrayMenu()
    {
        return new ContextMenuStrip
        {
            BackColor = Color.FromArgb(255, 255, 255),
            ForeColor = Color.FromArgb(31, 35, 40),
            Font = new Font("Segoe UI", 10F, FontStyle.Regular),
            Padding = new Padding(8, 9, 8, 9),
            ShowImageMargin = false,
            Renderer = new MutedTrayMenuRenderer()
        };
    }

    private static void AddHeader(ContextMenuStrip menu, string text)
    {
        var item = new ToolStripMenuItem(text)
        {
            Enabled = false,
            Font = new Font("Segoe UI Semibold", 10F, FontStyle.Bold),
            Padding = new Padding(12, 8, 18, 8),
            Margin = new Padding(3, 1, 3, 3)
        };
        menu.Items.Add(item);
        AddSeparator(menu);
    }

    private static void AddTrayItem(ContextMenuStrip menu, string text, EventHandler onClick)
    {
        var item = new ToolStripMenuItem(text)
        {
            Padding = new Padding(12, 7, 18, 7),
            Margin = new Padding(3, 1, 3, 1)
        };
        item.Click += onClick;
        menu.Items.Add(item);
    }

    private static void AddSeparator(ContextMenuStrip menu)
    {
        menu.Items.Add(new ToolStripSeparator { Margin = new Padding(4, 5, 4, 5) });
    }

    private static void ApplyRoundedRegion(ContextMenuStrip menu)
    {
        if (menu.Width <= 0 || menu.Height <= 0)
        {
            return;
        }
        using var path = RoundedRect(new Rectangle(0, 0, menu.Width, menu.Height), 8);
        menu.Region?.Dispose();
        menu.Region = new Region(path);
    }

    private static GraphicsPath RoundedRect(Rectangle rect, int radius)
    {
        var path = new GraphicsPath();
        var diameter = radius * 2;
        path.AddArc(rect.Left, rect.Top, diameter, diameter, 180, 90);
        path.AddArc(rect.Right - diameter - 1, rect.Top, diameter, diameter, 270, 90);
        path.AddArc(rect.Right - diameter - 1, rect.Bottom - diameter - 1, diameter, diameter, 0, 90);
        path.AddArc(rect.Left, rect.Bottom - diameter - 1, diameter, diameter, 90, 90);
        path.CloseFigure();
        return path;
    }

    private static Icon LoadTrayIcon()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "Assets", "ragflow.ico"),
            Path.Combine(FindRoot(), "launcher", "LocalMathRAGFlow", "Assets", "ragflow.ico"),
            Path.Combine(FindRoot(), "Assets", "ragflow.ico")
        };
        var path = candidates.FirstOrDefault(File.Exists);
        if (path is not null)
        {
            try
            {
                return new Icon(path);
            }
            catch
            {
            }
        }
        try
        {
            return Icon.ExtractAssociatedIcon(Application.ExecutablePath) ?? SystemIcons.Application;
        }
        catch
        {
            return SystemIcons.Application;
        }
    }

    public static void LogStartup(string message)
    {
        var logFile = Path.Combine(AppContext.BaseDirectory, "launcher-startup.log");
        LogTo(logFile, message);
    }

    private void Log(string message) => LogTo(logFile, message);

    private void OpenWeb(bool forceStatusPage = false)
    {
        var now = DateTimeOffset.UtcNow;
        if (!forceStatusPage && now - lastOpenAt < TimeSpan.FromMilliseconds(750))
        {
            return;
        }
        lastOpenAt = now;
        Log("open web requested");
        var url = StartupStatusServer.EntryUrl;
        var state = LoadBrowserWindowState();
        var browser = FindChromiumBrowser();
        if (browser is null)
        {
            Process.Start(new ProcessStartInfo { FileName = url, UseShellExecute = true });
            return;
        }

        if (forceStatusPage)
        {
            CloseBrowserProcess();
            browserProcess = null;
        }

        if (browserProcess is { HasExited: false })
        {
            browserProcess.Refresh();
            if (browserProcess.MainWindowHandle != IntPtr.Zero)
            {
                ApplyBrowserWindowState(browserProcess.MainWindowHandle, state);
                SetForegroundWindow(browserProcess.MainWindowHandle);
                return;
            }
        }

        var launcherDataDir = Path.Combine(FindRoot(), "data", "launcher");
        var profileDir = Path.Combine(
            launcherDataDir,
            $"browser-profile-run-{Environment.ProcessId}-{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}");
        Directory.CreateDirectory(profileDir);
        CleanupOldBrowserProfiles(launcherDataDir);
        Log($"starting browser app with profile {profileDir}");
        browserProcess = Process.Start(new ProcessStartInfo
        {
            FileName = browser,
            Arguments = $"--app=\"{url}\" --user-data-dir=\"{profileDir}\" --no-first-run{(state.Maximized ? " --start-maximized" : "")}",
            UseShellExecute = false
        });
    }

    private static void CleanupOldBrowserProfiles(string launcherDataDir)
    {
        try
        {
            foreach (var profile in Directory.EnumerateDirectories(launcherDataDir, "browser-profile-run-*"))
            {
                try
                {
                    var age = DateTimeOffset.Now - Directory.GetLastWriteTime(profile);
                    if (age > TimeSpan.FromDays(2))
                    {
                        Directory.Delete(profile, recursive: true);
                    }
                }
                catch
                {
                    // Old browser profiles can be locked by a still-running Edge process.
                }
            }
        }
        catch
        {
        }
    }

    private static BrowserWindowState LoadBrowserWindowState()
    {
        var path = BrowserWindowStatePath();
        if (!File.Exists(path))
        {
            return new BrowserWindowState(true);
        }

        try
        {
            return JsonSerializer.Deserialize<BrowserWindowState?>(File.ReadAllText(path, Encoding.UTF8)) ?? new BrowserWindowState(true);
        }
        catch
        {
            return new BrowserWindowState(true);
        }
    }

    private static string BrowserWindowStatePath()
    {
        return Path.Combine(FindRoot(), "data", "launcher", "window-state.json");
    }

    private void SaveCurrentBrowserWindowState()
    {
        if (browserProcess is not { HasExited: false })
        {
            return;
        }

        browserProcess.Refresh();
        var handle = browserProcess.MainWindowHandle;
        if (handle == IntPtr.Zero || !TryGetWindowPlacement(handle, out var placement))
        {
            return;
        }

        if (placement.ShowCmd == ShowWindowMinimized)
        {
            return;
        }

        var path = BrowserWindowStatePath();
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path) ?? FindRoot());
            File.WriteAllText(path, JsonSerializer.Serialize(new BrowserWindowState(placement.ShowCmd == ShowWindowMaximized)), Encoding.UTF8);
        }
        catch
        {
        }
    }

    private static void ApplyBrowserWindowState(IntPtr handle, BrowserWindowState state)
    {
        ShowWindow(handle, state.Maximized ? ShowWindowMaximized : ShowWindowRestore);
    }

    private static bool TryGetWindowPlacement(IntPtr handle, out WindowPlacement placement)
    {
        placement = new WindowPlacement { Length = Marshal.SizeOf<WindowPlacement>() };
        return GetWindowPlacement(handle, ref placement);
    }

    private async Task ExitAsync()
    {
        Log("menu exit clicked");
        OpenWeb(forceStatusPage: true);
        await BackgroundStartupTask.StopServicesAsync(exitAfterStop: true);
        CloseBrowserProcess();
        tray?.Dispose();
        ExitThread();
    }

    private void CloseBrowserProcess()
    {
        try
        {
            if (browserProcess is { HasExited: false })
            {
                browserProcess.Kill(entireProcessTree: true);
            }
        }
        catch
        {
        }
    }

    private static void OpenPath(string path)
    {
        if (!File.Exists(path) && !Directory.Exists(path))
        {
            if (Path.HasExtension(path))
            {
                Directory.CreateDirectory(Path.GetDirectoryName(path) ?? path);
                File.WriteAllText(path, "", Encoding.UTF8);
            }
            else
            {
                Directory.CreateDirectory(path);
            }
        }
        Process.Start(new ProcessStartInfo { FileName = path, UseShellExecute = true });
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

    private static int GetRagflowWebPort()
    {
        var envPath = Path.Combine(FindRoot(), "third_party", "ragflow", "docker", ".env");
        if (!File.Exists(envPath))
        {
            return 80;
        }
        foreach (var line in File.ReadLines(envPath))
        {
            if (line.StartsWith("SVR_WEB_HTTP_PORT=", StringComparison.Ordinal) &&
                int.TryParse(line.Split('=', 2)[1].Trim(), out var port))
            {
                return port;
            }
        }
        return 80;
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

    private static void LogTo(string file, string message)
    {
        try
        {
            File.AppendAllText(file, $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {message}\n", Encoding.UTF8);
        }
        catch
        {
        }
    }

    [DllImport("user32.dll")]
    private static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll")]
    private static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

    [DllImport("user32.dll")]
    private static extern bool GetWindowPlacement(IntPtr hWnd, ref WindowPlacement lpwndpl);

    private const int ShowWindowRestore = 9;
    private const int ShowWindowMaximized = 3;
    private const int ShowWindowMinimized = 2;

    private readonly record struct BrowserWindowState(bool Maximized);

    [StructLayout(LayoutKind.Sequential)]
    private struct WindowPlacement
    {
        public int Length;
        public int Flags;
        public int ShowCmd;
        public Point MinPosition;
        public Point MaxPosition;
        public Rectangle NormalPosition;
    }
}

internal static class StartupStatusServer
{
    public const int Port = 18765;
    public static string EntryUrl => $"http://127.0.0.1:{Port}/";
    private static int started;
    private static TcpListener? listener;

    public static void StartDetached()
    {
        if (Interlocked.Exchange(ref started, 1) == 1)
        {
            return;
        }

        try
        {
            listener = new TcpListener(IPAddress.Loopback, Port);
            listener.Start();
        }
        catch (Exception ex)
        {
            BackgroundStartupTask.WriteStatus("status_server_error", ex.Message);
            return;
        }

        _ = Task.Run(async () =>
        {
            try
            {
                await RunAsync(CancellationToken.None);
            }
            catch (Exception ex)
            {
                BackgroundStartupTask.WriteStatus("status_server_error", ex.Message);
            }
        });
    }

    private static async Task RunAsync(CancellationToken token)
    {
        if (listener is null)
        {
            return;
        }
        while (!token.IsCancellationRequested)
        {
            var client = await listener.AcceptTcpClientAsync(token);
            _ = Task.Run(async () => await HandleClientAsync(client), token);
        }
    }

    private static async Task HandleClientAsync(TcpClient client)
    {
        using var _client = client;
        using var stream = client.GetStream();
        using var reader = new StreamReader(stream, Encoding.ASCII, leaveOpen: true);
        var requestLine = await reader.ReadLineAsync() ?? "";
        while (!string.IsNullOrEmpty(await reader.ReadLineAsync()))
        {
        }

        var path = requestLine.Split(' ', StringSplitOptions.RemoveEmptyEntries).Skip(1).FirstOrDefault() ?? "/";
        if (path.StartsWith("/start", StringComparison.OrdinalIgnoreCase))
        {
            BackgroundStartupTask.StartDetached();
            await WriteResponseAsync(stream, "application/json; charset=utf-8", "{\"ok\":true}");
            return;
        }
        if (path.StartsWith("/stop", StringComparison.OrdinalIgnoreCase))
        {
            _ = Task.Run(async () => await BackgroundStartupTask.StopServicesAsync());
            await WriteResponseAsync(stream, "application/json; charset=utf-8", "{\"ok\":true}");
            return;
        }
        if (path.StartsWith("/status", StringComparison.OrdinalIgnoreCase))
        {
            await WriteResponseAsync(stream, "application/json; charset=utf-8", BuildStatusJson());
            return;
        }

        await WriteResponseAsync(stream, "text/html; charset=utf-8", BuildLoadingHtmlPage());
    }

    private static async Task WriteResponseAsync(NetworkStream stream, string contentType, string body)
    {
        var bodyBytes = Encoding.UTF8.GetBytes(body);
        var header = Encoding.ASCII.GetBytes(
            "HTTP/1.1 200 OK\r\n" +
            $"Content-Type: {contentType}\r\n" +
            "Cache-Control: no-store\r\n" +
            $"Content-Length: {bodyBytes.Length}\r\n" +
            "Connection: close\r\n\r\n");
        await stream.WriteAsync(header);
        await stream.WriteAsync(bodyBytes);
    }

    private static string BuildStatusJson()
    {
        var root = FindRoot();
        var statusPath = Path.Combine(root, "data", "launcher", "startup-status.json");
        var webUrl = $"http://127.0.0.1:{GetRagflowWebPort(root)}";
        var state = "starting";
        var message = "Starting LocalMathRAGFlow.";
        var updatedAt = DateTimeOffset.Now.ToString("O");

        if (File.Exists(statusPath))
        {
            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(statusPath, Encoding.UTF8));
                var rootElement = doc.RootElement;
                state = rootElement.TryGetProperty("state", out var stateValue) ? stateValue.GetString() ?? state : state;
                message = rootElement.TryGetProperty("message", out var messageValue) ? messageValue.GetString() ?? message : message;
                updatedAt = rootElement.TryGetProperty("updated_at", out var updatedValue) ? updatedValue.GetString() ?? updatedAt : updatedAt;
            }
            catch
            {
            }
        }

        var payload = new
        {
            state,
            message,
            updated_at = updatedAt,
            ready = state is "running",
            stopped = state is "stopped",
            close = state is "stopped_exit",
            web_url = $"{webUrl}/?localmathrag_reload={DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}"
        };
        return JsonSerializer.Serialize(payload);
    }

    private static string BuildLoadingHtmlPage()
    {
        return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LocalMathRAGFlow</title>
  <style>
    :root { color-scheme: light; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; color: #1f2328; background: #f6f8fa; }
    body { margin: 0; min-height: 100vh; display: grid; place-items: center; }
    main { width: min(520px, calc(100vw - 40px)); background: #fff; border: 1px solid #d8dee4; border-radius: 14px; padding: 26px 28px; box-shadow: 0 18px 48px rgba(31,35,40,.08); }
    h1 { margin: 0 0 8px; font-size: 22px; font-weight: 650; letter-spacing: 0; }
    p { margin: 0; color: #59636e; line-height: 1.6; }
    button { margin-top: 18px; border: 1px solid #0969da; border-radius: 8px; background: #0969da; color: #fff; font: inherit; font-weight: 600; padding: 9px 14px; cursor: pointer; }
    button[hidden] { display: none; }
    .bar { margin: 22px 0 14px; height: 8px; overflow: hidden; border-radius: 999px; background: #eaeef2; }
    .bar::before { content: ""; display: block; width: 42%; height: 100%; border-radius: inherit; background: #0969da; animation: move 1.2s ease-in-out infinite; }
    .meta { margin-top: 16px; font-size: 13px; color: #6e7781; }
    @keyframes move { 0% { transform: translateX(-100%); } 50% { transform: translateX(110%); } 100% { transform: translateX(250%); } }
  </style>
</head>
<body>
  <main>
    <h1>LocalMathRAGFlow &#27491;&#22312;&#21551;&#21160;</h1>
    <p id="message">&#27491;&#22312;&#26816;&#26597;&#21518;&#21488;&#26381;&#21153;&#12290;</p>
    <div class="bar" id="bar" aria-hidden="true"></div>
    <p class="meta" id="state">starting</p>
    <button id="start" hidden>Start service</button>
  </main>
  <script>
    const startButton = document.getElementById('start');
    const title = document.querySelector('h1');
    const bar = document.getElementById('bar');
    startButton.addEventListener('click', async () => {
      startButton.hidden = true;
      bar.classList.remove('stopped');
      title.textContent = 'LocalMathRAGFlow \u6b63\u5728\u542f\u52a8';
      document.getElementById('message').textContent = '\u6b63\u5728\u542f\u52a8\u540e\u53f0\u670d\u52a1\u3002';
      await fetch('/start', { method: 'POST' });
      tick();
    });
    async function tick() {
      let data;
      try {
        const res = await fetch('/status', { cache: 'no-store' });
        data = await res.json();
      } catch (e) {
        document.getElementById('message').textContent = '\u6b63\u5728\u7b49\u5f85\u542f\u52a8\u5668\u72b6\u6001\u670d\u52a1\u3002';
        return;
      }
      if (data.state === 'stopping') {
        title.textContent = 'LocalMathRAGFlow \u6b63\u5728\u505c\u6b62';
        bar.classList.remove('stopped');
      } else if (data.stopped) {
        title.textContent = 'LocalMathRAGFlow \u5df2\u505c\u6b62';
        bar.classList.add('stopped');
      } else {
        title.textContent = 'LocalMathRAGFlow \u6b63\u5728\u542f\u52a8';
        bar.classList.remove('stopped');
      }
      document.getElementById('message').textContent = data.message || '\u6b63\u5728\u542f\u52a8\u540e\u53f0\u670d\u52a1\u3002';
      document.getElementById('state').textContent = data.state || 'starting';
      startButton.hidden = !data.stopped || data.state === 'stopping';
      if (data.close) window.close();
      if (data.ready && data.web_url) window.location.replace(data.web_url);
    }
    tick();
    setInterval(tick, 1200);
  </script>
</body>
</html>
""";
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
        return installedRoot ?? candidates.FirstOrDefault() ?? Directory.GetCurrentDirectory();
    }

    private static int GetRagflowWebPort(string root)
    {
        var envPath = Path.Combine(root, "third_party", "ragflow", "docker", ".env");
        if (!File.Exists(envPath))
        {
            return 80;
        }
        foreach (var line in File.ReadLines(envPath))
        {
            if (line.StartsWith("SVR_WEB_HTTP_PORT=", StringComparison.Ordinal) &&
                int.TryParse(line.Split('=', 2)[1].Trim(), out var port))
            {
                return port;
            }
        }
        return 80;
    }
}

internal static class BackgroundStartupTask
{
    private static readonly object StatusLock = new();
    private static readonly object RunLock = new();
    private static CancellationTokenSource? runCts;
    private static Task? runTask;
    private static int stopRequested;

    public static void StartDetached()
    {
        lock (RunLock)
        {
            if (runTask is { IsCompleted: false })
            {
                return;
            }
            runCts?.Dispose();
            Interlocked.Exchange(ref stopRequested, 0);
            runCts = new CancellationTokenSource();
            var token = runCts.Token;
            runTask = Task.Run(async () =>
            {
                try
                {
                    await RunAsync(token);
                }
                catch (OperationCanceledException)
                {
                    if (Interlocked.CompareExchange(ref stopRequested, 0, 0) == 0)
                    {
                        WriteStatus("cancelled", "Startup task was cancelled.");
                    }
                }
                catch (Exception ex)
                {
                    WriteStatus("error", ex.Message);
                    Log("ERROR " + ex);
                }
            }, token);
        }
    }

    public static async Task StopServicesAsync(bool exitAfterStop = false)
    {
        CancellationTokenSource? cts;
        lock (RunLock)
        {
            cts = runCts;
        }
        Interlocked.Exchange(ref stopRequested, 1);
        cts?.Cancel();

        var root = FindRoot();
        WriteStatus("stopping", "Stopping Docker Compose services.");
        var script = Path.Combine(root, "scripts", "dev-down.ps1");
        if (!File.Exists(script))
        {
            WriteStatus("stop_failed", "dev-down.ps1 was not found.");
            return;
        }

        var exitCode = await RunProcessAsync(
            "powershell",
            $"-NoProfile -ExecutionPolicy Bypass -File \"{script}\"",
            root,
            CancellationToken.None);
        WriteStatus(exitCode == 0 ? (exitAfterStop ? "stopped_exit" : "stopped") : "stop_failed", exitCode == 0 ? "Background services stopped." : $"dev-down.ps1 exited with code {exitCode}.");
    }

    private static async Task RunAsync(CancellationToken token)
    {
        var root = FindRoot();
        Directory.CreateDirectory(Path.Combine(root, "data", "launcher"));
        var webUrl = $"http://127.0.0.1:{GetRagflowWebPort(root)}";
        WriteStatus("checking_service", "Checking whether RAGFlow Web is already running.");
        if (await WaitForRagflowWebReadyAsync(webUrl, TimeSpan.FromSeconds(3), token))
        {
            WriteStatus("running", "RAGFlow Web is already running.");
            return;
        }

        WriteStatus("checking_docker", "Checking Docker availability.");
        if (!CommandExists("docker"))
        {
            WriteStatus("docker_missing", "Docker CLI was not found. Install Docker Desktop and restart LocalMathRAGFlow.");
            return;
        }

        if (!await DockerReadyAsync(token))
        {
            WriteStatus("starting_docker", "Starting Docker Desktop.");
            StartDockerDesktop();
            if (!await WaitForDockerAsync(TimeSpan.FromMinutes(5), token))
            {
                WriteStatus("docker_timeout", "Docker Desktop did not become ready within 5 minutes.");
                return;
            }
        }

        var ragflowDocker = Path.Combine(root, "third_party", "ragflow", "docker");
        if (!Directory.Exists(ragflowDocker))
        {
            WriteStatus("ragflow_missing", "RAGFlow source is missing; user confirmation is required before downloading.");
            return;
        }

        WriteStatus("starting_compose", "Starting Docker Compose services.");
        var exitCode = await RunProcessAsync(
            "powershell",
            $"-NoProfile -ExecutionPolicy Bypass -File \"{Path.Combine(root, "scripts", "dev-up.ps1")}\"",
            root,
            token);
        if (exitCode != 0)
        {
            WriteStatus("compose_failed", $"dev-up.ps1 exited with code {exitCode}.");
            return;
        }

        WriteStatus("waiting_web", "Waiting for RAGFlow Web.");
        var webReady = await WaitForRagflowWebReadyAsync(webUrl, TimeSpan.FromMinutes(3), token);
        WriteStatus(webReady ? "running" : "web_warming", webReady ? "RAGFlow Web is ready." : "Containers are running; RAGFlow Web may still be warming up.");
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

    private static async Task<bool> DockerReadyAsync(CancellationToken token)
    {
        try
        {
            return await RunProcessAsync("docker", "info", FindRoot(), token) == 0;
        }
        catch
        {
            return false;
        }
    }

    private static async Task<bool> WaitForDockerAsync(TimeSpan timeout, CancellationToken token)
    {
        var start = DateTimeOffset.UtcNow;
        while (DateTimeOffset.UtcNow - start < timeout)
        {
            if (await DockerReadyAsync(token))
            {
                return true;
            }
            await Task.Delay(2000, token);
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
            }
            await Task.Delay(2000, token);
        }
        return false;
    }

    private static async Task<bool> WaitForRagflowWebReadyAsync(string webUrl, TimeSpan timeout, CancellationToken token)
    {
        using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
        var start = DateTimeOffset.UtcNow;
        while (DateTimeOffset.UtcNow - start < timeout)
        {
            try
            {
                if (!await IsHttpOkAsync(client, $"{webUrl}/api/v1/system/config", token))
                {
                    await Task.Delay(2000, token);
                    continue;
                }

                var indexHtml = await client.GetStringAsync($"{webUrl}/", token);
                if (!indexHtml.Contains("<div id=\"root\"></div>", StringComparison.OrdinalIgnoreCase))
                {
                    await Task.Delay(2000, token);
                    continue;
                }

                var assetsReady = true;
                foreach (var assetPath in ExtractStartupAssetPaths(indexHtml))
                {
                    if (!await IsHttpOkAsync(client, $"{webUrl}{assetPath}", token))
                    {
                        assetsReady = false;
                        break;
                    }
                }
                if (assetsReady)
                {
                    return true;
                }
            }
            catch
            {
            }
            await Task.Delay(2000, token);
        }
        return false;
    }

    private static async Task<bool> IsHttpOkAsync(HttpClient client, string url, CancellationToken token)
    {
        using var response = await client.GetAsync(url, token);
        return response.IsSuccessStatusCode;
    }

    private static IEnumerable<string> ExtractStartupAssetPaths(string html)
    {
        foreach (Match match in Regex.Matches(html, "(?:src|href)=\"(?<path>/(?:entry|chunk|assets)/[^\"]+)\"", RegexOptions.IgnoreCase))
        {
            yield return match.Groups["path"].Value;
        }
    }

    private static void StartDockerDesktop()
    {
        var candidates = new[]
        {
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "Docker", "Docker", "Docker Desktop.exe"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Docker", "Docker Desktop.exe")
        };
        var path = candidates.FirstOrDefault(File.Exists);
        if (path is null)
        {
            return;
        }
        Process.Start(new ProcessStartInfo
        {
            FileName = path,
            UseShellExecute = true,
            WindowStyle = ProcessWindowStyle.Hidden
        });
    }

    private static async Task<int> RunProcessAsync(string fileName, string arguments, string workingDirectory, CancellationToken token)
    {
        Log($"> {fileName} {arguments}");
        using var process = Process.Start(new ProcessStartInfo
        {
            FileName = fileName,
            Arguments = arguments,
            WorkingDirectory = workingDirectory,
            CreateNoWindow = true,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        });
        if (process is null)
        {
            return -1;
        }

        process.OutputDataReceived += (_, e) => { if (e.Data is not null) Log(e.Data); };
        process.ErrorDataReceived += (_, e) => { if (e.Data is not null) Log(e.Data); };
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        using var registration = token.Register(() =>
        {
            try
            {
                if (!process.HasExited)
                {
                    process.Kill(entireProcessTree: true);
                }
            }
            catch
            {
            }
        });
        await process.WaitForExitAsync(token);
        Log($"< exit {process.ExitCode}");
        return process.ExitCode;
    }

    private static void OpenUrl(string url)
    {
        try
        {
            Process.Start(new ProcessStartInfo { FileName = url, UseShellExecute = true });
        }
        catch (Exception ex)
        {
            Log("OPEN URL ERROR " + ex.Message);
        }
    }

    private static int GetRagflowWebPort(string root)
    {
        var envPath = Path.Combine(root, "third_party", "ragflow", "docker", ".env");
        if (!File.Exists(envPath))
        {
            return 80;
        }
        foreach (var line in File.ReadLines(envPath))
        {
            if (line.StartsWith("SVR_WEB_HTTP_PORT=", StringComparison.Ordinal) &&
                int.TryParse(line.Split('=', 2)[1].Trim(), out var port))
            {
                return port;
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

    public static void WriteStatus(string state, string message)
    {
        var root = FindRoot();
        var statusPath = Path.Combine(root, "data", "launcher", "startup-status.json");
        var payload = new
        {
            state,
            message,
            updated_at = DateTimeOffset.Now
        };
        lock (StatusLock)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(statusPath) ?? root);
            File.WriteAllText(statusPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }), Encoding.UTF8);
        }
        Log($"{state}: {message}");
    }

    private static void Log(string message)
    {
        try
        {
            var root = FindRoot();
            var logPath = Path.Combine(root, "data", "launcher", "startup.log");
            Directory.CreateDirectory(Path.GetDirectoryName(logPath) ?? root);
            File.AppendAllText(logPath, $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {message}\n", Encoding.UTF8);
        }
        catch
        {
        }
    }
}


internal sealed class MutedTrayMenuRenderer : ToolStripProfessionalRenderer
{
    private static readonly Color Back = Color.FromArgb(255, 255, 255);
    private static readonly Color Border = Color.FromArgb(218, 223, 230);
    private static readonly Color Hover = Color.FromArgb(244, 246, 248);
    private static readonly Color Pressed = Color.FromArgb(235, 238, 242);
    private static readonly Color ImageMargin = Color.FromArgb(255, 255, 255);

    public MutedTrayMenuRenderer() : base(new MutedColorTable())
    {
        RoundedEdges = true;
    }

    protected override void OnRenderToolStripBackground(ToolStripRenderEventArgs e)
    {
        using var brush = new SolidBrush(Back);
        e.Graphics.FillRectangle(brush, e.AffectedBounds);
    }

    protected override void OnRenderToolStripBorder(ToolStripRenderEventArgs e)
    {
        e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
        using var pen = new Pen(Border);
        var rect = new Rectangle(0, 0, e.ToolStrip.Width - 1, e.ToolStrip.Height - 1);
        using var path = RoundedRect(rect, 8);
        e.Graphics.DrawPath(pen, path);
    }

    protected override void OnRenderImageMargin(ToolStripRenderEventArgs e)
    {
        using var brush = new SolidBrush(ImageMargin);
        e.Graphics.FillRectangle(brush, e.AffectedBounds);
    }

    protected override void OnRenderMenuItemBackground(ToolStripItemRenderEventArgs e)
    {
        if (e.Item is not ToolStripMenuItem item || !item.Selected || !item.Enabled)
        {
            return;
        }

        var bounds = new Rectangle(5, 3, e.Item.Width - 10, e.Item.Height - 6);
        e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
        using var path = RoundedRect(bounds, 7);
        using var brush = new SolidBrush(item.Pressed ? Pressed : Hover);
        e.Graphics.FillPath(brush, path);
    }

    protected override void OnRenderSeparator(ToolStripSeparatorRenderEventArgs e)
    {
        using var pen = new Pen(Border);
        var y = e.Item.Height / 2;
        e.Graphics.DrawLine(pen, 12, y, e.Item.Width - 12, y);
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

    private sealed class MutedColorTable : ProfessionalColorTable
    {
        public override Color ToolStripDropDownBackground => Back;
        public override Color MenuBorder => Border;
        public override Color ImageMarginGradientBegin => ImageMargin;
        public override Color ImageMarginGradientMiddle => ImageMargin;
        public override Color ImageMarginGradientEnd => ImageMargin;
        public override Color MenuItemSelected => Hover;
        public override Color MenuItemBorder => Hover;
        public override Color MenuItemPressedGradientBegin => Pressed;
        public override Color MenuItemPressedGradientMiddle => Pressed;
        public override Color MenuItemPressedGradientEnd => Pressed;
    }
}
