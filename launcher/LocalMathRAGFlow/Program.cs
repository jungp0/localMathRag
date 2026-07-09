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
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using System.Windows.Forms;

namespace LocalMathRAGFlow;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
        if (LauncherInstanceCoordinator.TryForwardToExistingInstance())
        {
            return;
        }

        ApplicationConfiguration.Initialize();
        Application.ThreadException += (_, e) => TrayContext.LogStartup("THREAD ERROR " + e.Exception);
        AppDomain.CurrentDomain.UnhandledException += (_, e) => TrayContext.LogStartup("UNHANDLED ERROR " + e.ExceptionObject);
        Application.ApplicationExit += (_, _) => TrayContext.LogStartup("APPLICATION EXIT");
        StartupStatusServer.StartDetached();
        BackgroundStartupTask.StartDetached();
        Application.Run(new TrayContext());
    }
}

internal static class LauncherInstanceCoordinator
{
    public static string CurrentVersion
    {
        get
        {
            var path = Environment.ProcessPath ?? Application.ExecutablePath;
            var info = FileVersionInfo.GetVersionInfo(path);
            var fileVersion = info.ProductVersion;
            if (string.IsNullOrWhiteSpace(fileVersion))
            {
                fileVersion = info.FileVersion;
            }
            if (string.IsNullOrWhiteSpace(fileVersion))
            {
                fileVersion = typeof(Program).Assembly.GetName().Version?.ToString() ?? "0.0.0";
            }
            var buildTicks = File.Exists(path) ? File.GetLastWriteTimeUtc(path).Ticks : 0;
            return $"{fileVersion}+{buildTicks}";
        }
    }

    public static bool TryForwardToExistingInstance()
    {
        try
        {
            using var client = new HttpClient { Timeout = TimeSpan.FromMilliseconds(700) };
            var existingVersion = client.GetStringAsync($"{StartupStatusServer.EntryUrl}version").GetAwaiter().GetResult().Trim();
            if (string.IsNullOrWhiteSpace(existingVersion))
            {
                return false;
            }

            if (string.Equals(existingVersion, CurrentVersion, StringComparison.Ordinal))
            {
                client.PostAsync($"{StartupStatusServer.EntryUrl}focus", null).GetAwaiter().GetResult().Dispose();
                return true;
            }

            client.PostAsync($"{StartupStatusServer.EntryUrl}shutdown", null).GetAwaiter().GetResult().Dispose();
            WaitForPreviousInstanceToClose();
            return false;
        }
        catch
        {
            return false;
        }
    }

    private static void WaitForPreviousInstanceToClose()
    {
        var start = DateTimeOffset.UtcNow;
        while (DateTimeOffset.UtcNow - start < TimeSpan.FromSeconds(8))
        {
            try
            {
                using var client = new HttpClient { Timeout = TimeSpan.FromMilliseconds(300) };
                client.GetStringAsync($"{StartupStatusServer.EntryUrl}version").GetAwaiter().GetResult();
                Thread.Sleep(250);
            }
            catch
            {
                return;
            }
        }
    }
}

internal sealed class TrayContext : ApplicationContext
{
    private NotifyIcon? tray;
    private readonly string logFile;
    private readonly System.Windows.Forms.Timer windowStateTimer;
    private readonly SynchronizationContext uiContext;
    private Process? browserProcess;
    private DateTimeOffset lastOpenAt = DateTimeOffset.MinValue;

    public TrayContext()
    {
        logFile = Path.Combine(AppContext.BaseDirectory, "launcher-tray.log");
        Log("launcher tray starting");
        uiContext = SynchronizationContext.Current ?? new WindowsFormsSynchronizationContext();
        StartupStatusServer.RegisterControlHandlers(
            () => uiContext.Post(_ => OpenWeb(), null),
            () => uiContext.Post(_ => ShutdownLauncherOnly(), null));
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
        AddTrayItem(menu, "Reset runtime policy", async (_, _) =>
        {
            OpenWeb(forceStatusPage: true);
            await BackgroundStartupTask.ResetRuntimePolicyAsync();
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

        var launcherDataDir = Path.Combine(FindRoot(), "data", "launcher");
        var profileDir = Path.Combine(launcherDataDir, "browser-profile");
        Directory.CreateDirectory(profileDir);
        if (!forceStatusPage && TryFocusExistingBrowserWindow(state))
        {
            return;
        }

        CleanupOldBrowserProfiles(launcherDataDir);
        Log($"starting browser app with profile {profileDir}");
        var browserArgs = BuildBrowserArguments(url, profileDir, state);
        browserProcess = Process.Start(new ProcessStartInfo
        {
            FileName = browser,
            Arguments = browserArgs,
            UseShellExecute = false
        });
    }

    private bool TryFocusExistingBrowserWindow(BrowserWindowState state)
    {
        if (TryFocusBrowserProcessWindow(browserProcess, state, TimeSpan.FromSeconds(2)))
        {
            return true;
        }

        if (browserProcess is { HasExited: false })
        {
            Log("browser process is running without a visible window yet");
            return true;
        }

        if (TryFindBrowserAppWindow(out var handle))
        {
            ApplyBrowserWindowState(handle, state);
            SetForegroundWindow(handle);
            Log("focused existing browser app window");
            return true;
        }

        return false;
    }

    private bool TryFocusBrowserProcessWindow(Process? process, BrowserWindowState state, TimeSpan timeout)
    {
        if (process is null)
        {
            return false;
        }

        var start = DateTimeOffset.UtcNow;
        while (DateTimeOffset.UtcNow - start <= timeout)
        {
            try
            {
                if (process.HasExited)
                {
                    return false;
                }

                process.Refresh();
                var handle = process.MainWindowHandle;
                if (handle == IntPtr.Zero)
                {
                    handle = FindTopLevelWindowForProcess(process.Id);
                }
                if (handle != IntPtr.Zero)
                {
                    ApplyBrowserWindowState(handle, state);
                    SetForegroundWindow(handle);
                    Log("focused tracked browser app window");
                    return true;
                }
            }
            catch
            {
                return false;
            }
            Thread.Sleep(100);
        }

        return false;
    }

    private static string BuildBrowserArguments(string url, string profileDir, BrowserWindowState state)
    {
        var args = new List<string>
        {
            $"--app={QuoteArg(url)}",
            $"--user-data-dir={QuoteArg(profileDir)}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-sync",
            "--disable-component-update",
            "--disable-background-networking",
            "--disable-notifications",
            "--disable-features=msEdgeOnRampFRE,msEdgeSignIn,msEdgeSync,msEdgeWelcomePage,EdgeShoppingAssistant,msDiscoverChatButton"
        };
        if (state.Maximized)
        {
            args.Add("--start-maximized");
        }
        return string.Join(" ", args);
    }

    private static string QuoteArg(string value)
    {
        return "\"" + value.Replace("\"", "\\\"") + "\"";
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

    private static bool TryFindBrowserAppWindow(out IntPtr handle)
    {
        var foundHandle = IntPtr.Zero;
        EnumWindows((candidate, _) =>
        {
            if (!IsWindowVisible(candidate))
            {
                return true;
            }

            GetWindowThreadProcessId(candidate, out var processId);
            if (!IsChromiumProcess(processId))
            {
                return true;
            }

            var title = GetWindowTitle(candidate);
            if (!IsLocalMathRagFlowWindowTitle(title))
            {
                return true;
            }

            foundHandle = candidate;
            return false;
        }, IntPtr.Zero);
        handle = foundHandle;
        return handle != IntPtr.Zero;
    }

    private static IntPtr FindTopLevelWindowForProcess(int processId)
    {
        var handle = IntPtr.Zero;
        EnumWindows((candidate, _) =>
        {
            if (!IsWindowVisible(candidate))
            {
                return true;
            }

            GetWindowThreadProcessId(candidate, out var candidateProcessId);
            if (candidateProcessId != processId)
            {
                return true;
            }

            handle = candidate;
            return false;
        }, IntPtr.Zero);
        return handle;
    }

    private static bool IsChromiumProcess(uint processId)
    {
        try
        {
            using var process = Process.GetProcessById(unchecked((int)processId));
            return process.ProcessName.Equals("msedge", StringComparison.OrdinalIgnoreCase) ||
                   process.ProcessName.Equals("chrome", StringComparison.OrdinalIgnoreCase);
        }
        catch
        {
            return false;
        }
    }

    private static string GetWindowTitle(IntPtr handle)
    {
        var length = GetWindowTextLength(handle);
        if (length <= 0)
        {
            return "";
        }

        var builder = new StringBuilder(length + 1);
        GetWindowText(handle, builder, builder.Capacity);
        return builder.ToString();
    }

    private static bool IsLocalMathRagFlowWindowTitle(string title)
    {
        return title.Contains("LocalMathRAGFlow", StringComparison.OrdinalIgnoreCase) ||
               title.Contains("RAGFlow", StringComparison.OrdinalIgnoreCase);
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

    private void ShutdownLauncherOnly()
    {
        Log("shutdown requested by newer launcher");
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

    [DllImport("user32.dll")]
    private static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);

    [DllImport("user32.dll")]
    private static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll")]
    private static extern int GetWindowTextLength(IntPtr hWnd);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

    [DllImport("user32.dll")]
    private static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);

    private const int ShowWindowRestore = 9;
    private const int ShowWindowMaximized = 3;
    private const int ShowWindowMinimized = 2;

    private delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

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
    private static Action? focusRequested;
    private static Action? shutdownRequested;

    public static void RegisterControlHandlers(Action onFocusRequested, Action onShutdownRequested)
    {
        focusRequested = onFocusRequested;
        shutdownRequested = onShutdownRequested;
    }

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
        if (path.StartsWith("/version", StringComparison.OrdinalIgnoreCase))
        {
            await WriteResponseAsync(stream, "text/plain; charset=utf-8", LauncherInstanceCoordinator.CurrentVersion);
            return;
        }
        if (path.StartsWith("/focus", StringComparison.OrdinalIgnoreCase))
        {
            focusRequested?.Invoke();
            await WriteResponseAsync(stream, "application/json; charset=utf-8", "{\"ok\":true}");
            return;
        }
        if (path.StartsWith("/shutdown", StringComparison.OrdinalIgnoreCase))
        {
            shutdownRequested?.Invoke();
            await WriteResponseAsync(stream, "application/json; charset=utf-8", "{\"ok\":true}");
            return;
        }
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
        if (path.StartsWith("/reset-runtime-policy", StringComparison.OrdinalIgnoreCase))
        {
            _ = Task.Run(async () => await BackgroundStartupTask.ResetRuntimePolicyAsync());
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
            resetting = state is "resetting_policy",
            close = state is "stopped_exit",
            language = ResolveConfiguredLanguage(root),
            web_url = $"{webUrl}/?localmathrag_reload={DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}"
        };
        return JsonSerializer.Serialize(payload);
    }

    private static string BuildLoadingHtmlPage()
    {
        var root = FindRoot();
        var webUrl = $"http://127.0.0.1:{GetRagflowWebPort(root)}";
        var language = ResolveConfiguredLanguage(root);
        return """
<!doctype html>
<html lang="en" data-configured-language="__LANGUAGE__" data-web-url="__WEB_URL__">
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
    .actions { margin-top: 18px; display: grid; grid-template-columns: auto auto minmax(0, 1fr); align-items: center; gap: 10px 12px; }
    button { border: 1px solid #0969da; border-radius: 8px; background: #0969da; color: #fff; font: inherit; font-weight: 600; padding: 9px 14px; cursor: pointer; }
    button.secondary { border-color: #d0d7de; background: #fff; color: #24292f; }
    button[hidden] { display: none; }
    button:disabled { cursor: wait; opacity: .62; }
    .policy-note { min-width: 0; font-size: 13px; color: #59636e; line-height: 1.45; }
    @media (max-width: 560px) { .actions { grid-template-columns: 1fr; } .actions button { width: 100%; } }
    .bar { margin: 22px 0 14px; height: 8px; overflow: hidden; border-radius: 999px; background: #eaeef2; }
    .bar::before { content: ""; display: block; width: 42%; height: 100%; border-radius: inherit; background: #0969da; animation: move 1.2s ease-in-out infinite; }
    .meta { margin-top: 16px; font-size: 13px; color: #6e7781; }
    @keyframes move { 0% { transform: translateX(-100%); } 50% { transform: translateX(110%); } 100% { transform: translateX(250%); } }
  </style>
</head>
<body>
  <main>
    <h1 id="title">LocalMathRAGFlow</h1>
    <p id="message"></p>
    <div class="bar" id="bar" aria-hidden="true"></div>
    <p class="meta" id="state">starting</p>
    <div class="actions">
      <button id="start" hidden></button>
      <button id="reset" class="secondary"></button>
      <p class="policy-note" id="policy-note"></p>
    </div>
  </main>
  <script>
    const translations = {
      en: {
        startingTitle: 'LocalMathRAGFlow is starting',
        stoppingTitle: 'LocalMathRAGFlow is stopping',
        stoppedTitle: 'LocalMathRAGFlow is stopped',
        resettingTitle: 'LocalMathRAGFlow is resetting',
        startButton: 'Start service',
        resetButton: 'Reset runtime policy',
        policyNote: 'Local model calls use dynamic scheduling. The app generates a configuration from previous runtime results. If hardware changed or startup is stuck, use reset to clean resources and generate a fresh configuration.',
        confirmReset: 'Reset runtime policy, stop model containers, and reconfigure services now?',
        statusUnavailable: 'Waiting for the launcher status service.',
        statusUnavailableProbe: 'Waiting for the launcher status service. Checking whether RAGFlow is already available.',
        statusUnavailableStopped: 'The launcher status service is unavailable and RAGFlow is not responding. Background services appear to be stopped.',
        checking_service: 'Checking whether RAGFlow Web is already running.',
        checking_docker: 'Checking Docker availability.',
        starting_docker: 'Starting Docker Desktop.',
        starting_compose: 'Starting Docker Compose services.',
        waiting_web: 'Waiting for RAGFlow Web.',
        running: 'RAGFlow Web is ready.',
        web_warming: 'Containers are running; RAGFlow Web may still be warming up.',
        stopped: 'Background services stopped.',
        stopped_exit: 'Background services stopped.',
        stopping: 'Stopping Docker Compose services.',
        resetting_policy: 'Resetting runtime policy and cleaning model runtime resources.',
        docker_missing: 'Docker CLI was not found. Install Docker Desktop and restart LocalMathRAGFlow.',
        docker_timeout: 'Docker Desktop did not become ready within 5 minutes.',
        ragflow_missing: 'RAGFlow source is missing; user confirmation is required before downloading.',
        compose_failed: 'Docker Compose startup failed.',
        error: 'Startup failed.',
      },
      'zh-Hans': {
        startingTitle: 'LocalMathRAGFlow \u6b63\u5728\u542f\u52a8',
        stoppingTitle: 'LocalMathRAGFlow \u6b63\u5728\u505c\u6b62',
        stoppedTitle: 'LocalMathRAGFlow \u5df2\u505c\u6b62',
        resettingTitle: 'LocalMathRAGFlow \u6b63\u5728\u91cd\u7f6e',
        startButton: '\u542f\u52a8\u670d\u52a1',
        resetButton: '\u91cd\u7f6e\u8fd0\u884c\u7b56\u7565',
        policyNote: '\u672c\u5730\u6a21\u578b\u8c03\u7528\u91c7\u7528\u52a8\u6001\u8c03\u5ea6\uff0c\u6839\u636e\u5386\u53f2\u8fd0\u884c\u7ed3\u679c\u751f\u6210\u914d\u7f6e\u65b9\u6848\u3002\u914d\u7f6e\u53d8\u5316\u6216\u542f\u52a8\u5361\u4f4f\u65f6\uff0c\u53ef\u4ee5\u6309\u91cd\u7f6e\u6309\u94ae\u6e05\u7406\u8d44\u6e90\u5e76\u91cd\u65b0\u751f\u6210\u914d\u7f6e\u3002',
        confirmReset: '\u73b0\u5728\u91cd\u7f6e\u8fd0\u884c\u7b56\u7565\u3001\u505c\u6b62\u6a21\u578b\u5bb9\u5668\u5e76\u91cd\u65b0\u914d\u7f6e\u670d\u52a1\uff1f',
        statusUnavailable: '\u6b63\u5728\u7b49\u5f85\u542f\u52a8\u5668\u72b6\u6001\u670d\u52a1\u3002',
        statusUnavailableProbe: '\u6b63\u5728\u7b49\u5f85\u542f\u52a8\u5668\u72b6\u6001\u670d\u52a1\uff0c\u540c\u65f6\u68c0\u67e5 RAGFlow \u662f\u5426\u5df2\u53ef\u7528\u3002',
        statusUnavailableStopped: '\u542f\u52a8\u5668\u72b6\u6001\u670d\u52a1\u4e0d\u53ef\u7528\uff0cRAGFlow \u4e5f\u672a\u54cd\u5e94\u3002\u540e\u53f0\u670d\u52a1\u53ef\u80fd\u5df2\u505c\u6b62\u3002',
        checking_service: '\u6b63\u5728\u68c0\u67e5 RAGFlow Web \u662f\u5426\u5df2\u8fd0\u884c\u3002',
        checking_docker: '\u6b63\u5728\u68c0\u67e5 Docker \u53ef\u7528\u6027\u3002',
        starting_docker: '\u6b63\u5728\u542f\u52a8 Docker Desktop\u3002',
        starting_compose: '\u6b63\u5728\u542f\u52a8 Docker Compose \u670d\u52a1\u3002',
        waiting_web: '\u6b63\u5728\u7b49\u5f85 RAGFlow Web\u3002',
        running: 'RAGFlow Web \u5df2\u5c31\u7eea\u3002',
        web_warming: '\u5bb9\u5668\u5df2\u8fd0\u884c\uff0cRAGFlow Web \u53ef\u80fd\u4ecd\u5728\u9884\u70ed\u3002',
        stopped: '\u540e\u53f0\u670d\u52a1\u5df2\u505c\u6b62\u3002',
        stopped_exit: '\u540e\u53f0\u670d\u52a1\u5df2\u505c\u6b62\u3002',
        stopping: '\u6b63\u5728\u505c\u6b62 Docker Compose \u670d\u52a1\u3002',
        resetting_policy: '\u6b63\u5728\u91cd\u7f6e\u8fd0\u884c\u7b56\u7565\u5e76\u6e05\u7406\u6a21\u578b\u8fd0\u884c\u8d44\u6e90\u3002',
        docker_missing: '\u672a\u627e\u5230 Docker CLI\u3002\u8bf7\u5b89\u88c5 Docker Desktop \u540e\u91cd\u542f LocalMathRAGFlow\u3002',
        docker_timeout: 'Docker Desktop \u5728 5 \u5206\u949f\u5185\u672a\u5c31\u7eea\u3002',
        ragflow_missing: 'RAGFlow \u6e90\u7801\u7f3a\u5931\uff1b\u9700\u8981\u7528\u6237\u786e\u8ba4\u540e\u624d\u80fd\u4e0b\u8f7d\u3002',
        compose_failed: 'Docker Compose \u542f\u52a8\u5931\u8d25\u3002',
        error: '\u542f\u52a8\u5931\u8d25\u3002',
      },
    };
    const startButton = document.getElementById('start');
    const resetButton = document.getElementById('reset');
    const title = document.getElementById('title');
    const message = document.getElementById('message');
    const stateText = document.getElementById('state');
    const policyNote = document.getElementById('policy-note');
    const bar = document.getElementById('bar');
    const configuredWebUrl = document.documentElement.dataset.webUrl || 'http://127.0.0.1/';
    let currentLanguage = normalizeLanguage(
      localStorage.getItem('lng') ||
      document.documentElement.dataset.configuredLanguage ||
      navigator.language ||
      'en'
    );
    let statusFailures = 0;

    function normalizeLanguage(value) {
      const lng = String(value || '').toLowerCase();
      if (lng.startsWith('zh')) return 'zh-Hans';
      return 'en';
    }
    function tr(key) {
      return (translations[currentLanguage] && translations[currentLanguage][key]) || translations.en[key] || key;
    }
    function applyLanguage(lng) {
      currentLanguage = normalizeLanguage(lng || currentLanguage);
      document.documentElement.lang = currentLanguage === 'zh-Hans' ? 'zh-CN' : 'en';
      startButton.textContent = tr('startButton');
      resetButton.textContent = tr('resetButton');
      policyNote.textContent = tr('policyNote');
      if (!stateText.textContent || stateText.textContent === 'starting') {
        title.textContent = tr('startingTitle');
        message.textContent = tr('checking_service');
      }
    }
    function localizedMessage(data) {
      if (translations[currentLanguage] && translations[currentLanguage][data.state]) {
        return tr(data.state);
      }
      return data.message || tr('checking_service');
    }
    function renderStatusUnavailableStopped() {
      title.textContent = tr('stoppedTitle');
      message.textContent = tr('statusUnavailableStopped');
      stateText.textContent = 'stopped';
      bar.classList.add('stopped');
      startButton.hidden = true;
      resetButton.disabled = true;
    }
    async function probeRagflowWhenStatusIsUnavailable() {
      statusFailures += 1;
      message.textContent = statusFailures >= 2 ? tr('statusUnavailableProbe') : tr('statusUnavailable');
      if (statusFailures < 2) return;
      try {
        await fetch(configuredWebUrl, { mode: 'no-cors', cache: 'no-store' });
        window.location.replace(configuredWebUrl);
      } catch (e) {
      }
      if (statusFailures >= 5) {
        renderStatusUnavailableStopped();
        if (statusFailures >= 6) window.close();
      }
    }
    applyLanguage(currentLanguage);
    startButton.addEventListener('click', async () => {
      startButton.hidden = true;
      bar.classList.remove('stopped');
      title.textContent = tr('startingTitle');
      message.textContent = tr('starting_compose');
      await fetch('/start', { method: 'POST' });
      tick();
    });
    resetButton.addEventListener('click', async () => {
      if (!confirm(tr('confirmReset'))) return;
      resetButton.disabled = true;
      startButton.hidden = true;
      bar.classList.remove('stopped');
      title.textContent = tr('resettingTitle');
      message.textContent = tr('resetting_policy');
      await fetch('/reset-runtime-policy', { method: 'POST' });
      tick();
    });
    async function tick() {
      let data;
      try {
        const res = await fetch('/status', { cache: 'no-store' });
        data = await res.json();
      } catch (e) {
        await probeRagflowWhenStatusIsUnavailable();
        return;
      }
      statusFailures = 0;
      applyLanguage(data.language);
      if (data.state === 'stopping') {
        title.textContent = tr('stoppingTitle');
        bar.classList.remove('stopped');
      } else if (data.resetting) {
        title.textContent = tr('resettingTitle');
        bar.classList.remove('stopped');
      } else if (data.stopped) {
        title.textContent = tr('stoppedTitle');
        bar.classList.add('stopped');
      } else {
        title.textContent = tr('startingTitle');
        bar.classList.remove('stopped');
      }
      message.textContent = localizedMessage(data);
      stateText.textContent = data.state || 'starting';
      startButton.hidden = !data.stopped || data.state === 'stopping';
      resetButton.disabled = data.resetting || data.state === 'stopping';
      if (data.close) window.close();
      if (data.ready && data.web_url) window.location.replace(data.web_url);
    }
    tick();
    setInterval(() => {
      if (statusFailures < 6) tick();
    }, 1200);
  </script>
</body>
</html>
""".Replace("__LANGUAGE__", HtmlAttributeEncode(language))
            .Replace("__WEB_URL__", HtmlAttributeEncode($"{webUrl}/?localmathrag_reload={DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}"));
    }

    private static string HtmlAttributeEncode(string value)
    {
        return WebUtility.HtmlEncode(value).Replace("'", "&#39;");
    }

    private static string ResolveConfiguredLanguage(string root)
    {
        var candidates = new[]
        {
            Environment.GetEnvironmentVariable("LOCALMATHRAG_LANGUAGE"),
            Environment.GetEnvironmentVariable("VITE_DEFAULT_LANGUAGE_CODE"),
            ReadEnvValue(Path.Combine(root, "third_party", "ragflow", "web", ".env"), "VITE_DEFAULT_LANGUAGE_CODE"),
            ReadEnvValue(Path.Combine(root, "third_party", "ragflow", "web", ".env.production"), "VITE_DEFAULT_LANGUAGE_CODE"),
        };
        foreach (var candidate in candidates)
        {
            var normalized = NormalizeLanguage(candidate);
            if (!string.IsNullOrWhiteSpace(normalized))
            {
                return normalized;
            }
        }
        return "en";
    }

    private static string? ReadEnvValue(string path, string key)
    {
        if (!File.Exists(path))
        {
            return null;
        }
        foreach (var rawLine in File.ReadLines(path))
        {
            var line = rawLine.Trim();
            if (line.Length == 0 || line.StartsWith("#", StringComparison.Ordinal))
            {
                continue;
            }
            if (!line.StartsWith(key + "=", StringComparison.Ordinal))
            {
                continue;
            }
            var value = line.Split('=', 2)[1].Split('#', 2)[0].Trim();
            return value.Trim('\'', '"');
        }
        return null;
    }

    private static string? NormalizeLanguage(string? language)
    {
        if (string.IsNullOrWhiteSpace(language))
        {
            return null;
        }
        var normalized = language.Trim();
        return normalized.StartsWith("zh", StringComparison.OrdinalIgnoreCase) ? "zh-Hans" : "en";
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
    private static int dockerDesktopStartedByLauncher;

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
        if (exitCode == 0 && exitAfterStop)
        {
            await ReleaseDockerResourcesOnExitAsync(root);
        }
    }

    public static async Task ResetRuntimePolicyAsync()
    {
        CancellationTokenSource? cts;
        Task? previousTask;
        lock (RunLock)
        {
            cts = runCts;
            previousTask = runTask;
        }

        Interlocked.Exchange(ref stopRequested, 1);
        cts?.Cancel();
        if (previousTask is not null)
        {
            try
            {
                await previousTask;
            }
            catch
            {
            }
        }

        var root = FindRoot();
        WriteStatus("resetting_policy", "Resetting runtime policy and cleaning model runtime resources.");
        var downScript = Path.Combine(root, "scripts", "dev-down.ps1");
        if (File.Exists(downScript))
        {
            var stopExitCode = await RunProcessAsync(
                "powershell",
                $"-NoProfile -ExecutionPolicy Bypass -File \"{downScript}\"",
                root,
                CancellationToken.None);
            if (stopExitCode != 0)
            {
                WriteStatus("resetting_policy", $"dev-down.ps1 exited with code {stopExitCode}; resetting policy anyway.");
            }
        }

        ResetRuntimeConfigPolicy(root);
        WriteStatus("resetting_policy", "Runtime policy reset; reconfiguring services.");
        Interlocked.Exchange(ref stopRequested, 0);
        StartDetached();
    }

    private static void ResetRuntimeConfigPolicy(string root)
    {
        var configPath = Path.Combine(root, "data", "cache", "runtime-config.json");
        Directory.CreateDirectory(Path.GetDirectoryName(configPath) ?? root);
        JsonObject config;
        if (File.Exists(configPath))
        {
            try
            {
                config = JsonNode.Parse(File.ReadAllText(configPath, Encoding.UTF8)) as JsonObject ?? new JsonObject();
            }
            catch
            {
                config = new JsonObject();
            }
        }
        else
        {
            config = new JsonObject();
        }

        config.Remove("scheduler");
        config.Remove("rerank");
        config["version"] = 1;
        config["updated_at"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        File.WriteAllText(
            configPath,
            config.ToJsonString(new JsonSerializerOptions { WriteIndented = true }),
            Encoding.UTF8);
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
            if (StartDockerDesktop())
            {
                Interlocked.Exchange(ref dockerDesktopStartedByLauncher, 1);
            }
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

    private static bool StartDockerDesktop()
    {
        var candidates = new[]
        {
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "Docker", "Docker", "Docker Desktop.exe"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Docker", "Docker Desktop.exe")
        };
        var path = candidates.FirstOrDefault(File.Exists);
        if (path is null)
        {
            return false;
        }
        Process.Start(new ProcessStartInfo
        {
            FileName = path,
            UseShellExecute = true,
            WindowStyle = ProcessWindowStyle.Hidden
        });
        return true;
    }

    private static async Task ReleaseDockerResourcesOnExitAsync(string root)
    {
        if (!ShouldReleaseDockerWslOnExit() ||
            Interlocked.CompareExchange(ref dockerDesktopStartedByLauncher, 0, 0) == 0)
        {
            return;
        }

        if (await HasRunningDockerContainersAsync(root))
        {
            Log("Skipping Docker Desktop shutdown because other containers are still running.");
            return;
        }

        await ShutdownDockerDesktopAsync(root);
        await ShutdownWslAsync(root);
    }

    private static bool ShouldReleaseDockerWslOnExit()
    {
        var value = Environment.GetEnvironmentVariable("LOCALMATHRAG_RELEASE_DOCKER_WSL_ON_EXIT");
        return value is null || value.Equals("1", StringComparison.OrdinalIgnoreCase) ||
            value.Equals("true", StringComparison.OrdinalIgnoreCase) ||
            value.Equals("yes", StringComparison.OrdinalIgnoreCase) ||
            value.Equals("on", StringComparison.OrdinalIgnoreCase);
    }

    private static async Task<bool> HasRunningDockerContainersAsync(string root)
    {
        try
        {
            var (exitCode, output) = await RunProcessCaptureAsync("docker", "ps --quiet", root, TimeSpan.FromSeconds(10));
            return exitCode == 0 && !string.IsNullOrWhiteSpace(output);
        }
        catch (Exception ex)
        {
            Log("Docker container check failed before shutdown: " + ex.Message);
            return true;
        }
    }

    private static async Task ShutdownDockerDesktopAsync(string root)
    {
        var candidates = new[]
        {
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "Docker", "Docker", "DockerCli.exe"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Docker", "DockerCli.exe")
        };
        var dockerCli = candidates.FirstOrDefault(File.Exists);
        if (dockerCli is null)
        {
            return;
        }

        try
        {
            await RunProcessCaptureAsync(dockerCli, "-Shutdown", root, TimeSpan.FromSeconds(30));
        }
        catch (OperationCanceledException)
        {
            Log("Docker Desktop shutdown command timed out.");
        }
    }

    private static async Task ShutdownWslAsync(string root)
    {
        try
        {
            await RunProcessCaptureAsync("wsl.exe", "--shutdown", root, TimeSpan.FromSeconds(15));
        }
        catch (OperationCanceledException)
        {
            Log("WSL shutdown command timed out.");
        }
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

    private static async Task<(int ExitCode, string Output)> RunProcessCaptureAsync(string fileName, string arguments, string workingDirectory, TimeSpan timeout)
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
            return (-1, string.Empty);
        }

        var output = new StringBuilder();
        process.OutputDataReceived += (_, e) =>
        {
            if (e.Data is not null)
            {
                output.AppendLine(e.Data);
                Log(e.Data);
            }
        };
        process.ErrorDataReceived += (_, e) => { if (e.Data is not null) Log(e.Data); };
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        using var cts = new CancellationTokenSource(timeout);
        using var registration = cts.Token.Register(() =>
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
        await process.WaitForExitAsync(cts.Token);
        Log($"< exit {process.ExitCode}");
        return (process.ExitCode, output.ToString());
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
