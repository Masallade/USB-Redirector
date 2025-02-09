import os
import subprocess
import time
import win32gui
import win32con
import sys
import tempfile
import shutil
import tkinter as tk
import winreg

def get_resource_path(filename):
    if hasattr(sys, "_MEIPASS"):  # If running as a PyInstaller executable
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

USB_SERVER_EXE = "usbserver.exe"
USB_SERVER_INSTALLER = "usb-over-network-server-64bit.msi"

USB_SERVER_PATH = get_resource_path(USB_SERVER_EXE)
USB_SERVER_INSTALLER_PATH = get_resource_path(USB_SERVER_INSTALLER)

def is_usbserver_installed():
    """Check if the USB server is already installed using the registry."""
    try:
        reg_path = r"SOFTWARE\FabulaTech\USBRedirectorServer"
        reg_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path)
        winreg.CloseKey(reg_key)
        return True
    except FileNotFoundError:
        return False

def install_usbserver():
    """Run the USB server installer if not already installed."""
    if not is_usbserver_installed():
        print("USB Server not found. Installing...")
        try:
            subprocess.run(
                ["msiexec", "/i", USB_SERVER_INSTALLER_PATH, "/quiet"],
                check=True,
            )
            print("USB Server installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install USB Server: {e}")
    else:
        print("USB Server is already installed.")

def extract_usbserver():
    """Extract the USB server executable to a temporary directory."""
    temp_dir = os.path.join(tempfile.gettempdir(), "usbserver_temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_usbserver_path = os.path.join(temp_dir, USB_SERVER_EXE)
    if not os.path.exists(USB_SERVER_PATH):
        raise FileNotFoundError(f"{USB_SERVER_EXE} not found at {USB_SERVER_PATH}")
    shutil.copy2(USB_SERVER_PATH, temp_usbserver_path)
    return temp_usbserver_path

usb_server_path_temp = extract_usbserver()
usb_server_process = None
usb_server_hwnd = None

def find_window_by_title(title):
    """Find window handle by its title, retrying for 10 seconds."""
    hwnd = 0
    for _ in range(10):
        hwnd = win32gui.FindWindow(None, title)
        if hwnd:
            return hwnd
        time.sleep(1)
    return 0

def remove_title_bar(hwnd):
    """Remove title bar, borders, and menu of the window."""
    style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
    style &= ~(win32con.WS_CAPTION | win32con.WS_THICKFRAME |
               win32con.WS_MINIMIZEBOX | win32con.WS_MAXIMIZEBOX | win32con.WS_SYSMENU)
    win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)
    
    if win32gui.GetMenu(hwnd):
        win32gui.SetMenu(hwnd, None)
    
    def enum_child_proc(child_hwnd, _):
        class_name = win32gui.GetClassName(child_hwnd).lower()
        if "menu" in class_name or "toolbar" in class_name:
            win32gui.ShowWindow(child_hwnd, win32con.SW_HIDE)
    
    win32gui.EnumChildWindows(hwnd, enum_child_proc, None)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, 0, 0, 800, 600,
                          win32con.SWP_FRAMECHANGED | win32con.SWP_NOZORDER | win32con.SWP_SHOWWINDOW)

def resize_usbserver(event=None):
    """Resize and reposition USB server window to match Tkinter window."""
    global usb_server_hwnd
    if usb_server_hwnd:
        width = root.winfo_width()
        height = root.winfo_height()
        win32gui.SetWindowPos(usb_server_hwnd, win32con.HWND_TOP, 0, 50, width, height - 50,
                              win32con.SWP_NOZORDER | win32con.SWP_SHOWWINDOW)

def embed_usbserver():
    """Start USB server and embed it into the Tkinter window."""
    global usb_server_process, usb_server_hwnd
    
    try:
        usb_server_process = subprocess.Popen(usb_server_path_temp, creationflags=subprocess.DETACHED_PROCESS)
        usb_server_hwnd = find_window_by_title("USB over Network Server - www.fabulatech.com")
    except Exception as e:
        status_label.config(text=f"Error: {e}")
        return
    
    if not usb_server_hwnd:
        status_label.config(text="Failed to find USB server window!")
        return
    
    remove_title_bar(usb_server_hwnd)
    root.update()
    tk_window_hwnd = win32gui.FindWindow(None, "USB Redirector Host")
    if not tk_window_hwnd:
        status_label.config(text="Failed to get Tkinter window handle!")
        return
    
    win32gui.SetParent(usb_server_hwnd, tk_window_hwnd)
    resize_usbserver()
    status_label.config(text="USB Server embedded successfully!")

def on_closing():
    """Exit app and close USB server."""
    global usb_server_process
    if usb_server_process:
        usb_server_process.terminate()
    root.destroy()

def delete_shortcut_from_desktop(shortcut_name):
    user_desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    
    public_desktop_path = os.path.join(os.environ['PUBLIC'], 'Desktop')
    
    user_shortcut_path = os.path.join(user_desktop_path, shortcut_name + '.lnk')
    public_shortcut_path = os.path.join(public_desktop_path, shortcut_name + '.lnk')
    
    if os.path.exists(user_shortcut_path):
        os.remove(user_shortcut_path)
        print(f"Shortcut '{shortcut_name}' has been deleted from the user's desktop.")
    elif os.path.exists(public_shortcut_path):
        os.remove(public_shortcut_path)
        print(f"Shortcut '{shortcut_name}' has been deleted from the public desktop.")
    else:
        print(f"Shortcut '{shortcut_name}' does not exist on either desktop.")    


# Install USB server if needed
install_usbserver()

# Create Tkinter window
root = tk.Tk()
root.title("USB Redirector Host")
root.geometry("900x700")
root.configure(bg="#f0f0f0")

header_label = tk.Label(root, text="USB Redirector Host", font=("Arial", 18, "bold"), bg="#f0f0f0", fg="#333")
header_label.pack(pady=10)

status_label = tk.Label(root, text="Initializing...", font=("Arial", 12), bg="#f0f0f0", fg="#666")
status_label.pack(pady=5)

instructions_label = tk.Label(root, text="Resize the window to adjust the USB server display.", font=("Arial", 10), bg="#f0f0f0", fg="#666")
instructions_label.pack(pady=10)

root.after(2000, embed_usbserver)
root.bind("<Configure>", resize_usbserver)
root.protocol("WM_DELETE_WINDOW", on_closing)

delete_shortcut_from_desktop("USB over Network (Server)")


root.mainloop()
