#!/usr/bin/env bash
#
# OS-level performance tuning for a Raspberry Pi 5 running Marine AI.
#
#   sudo ./scripts/setup_pi_os.sh            # apply everything
#   sudo ./scripts/setup_pi_os.sh --dry-run  # show what would change
#   sudo ./scripts/setup_pi_os.sh --status   # report current state only
#
# Covers: ZRAM, CPU governor, USB power management, thermal reporting.
# Every change is idempotent and prints what it did.
#
set -euo pipefail

DRY_RUN=0
STATUS_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --status) STATUS_ONLY=1 ;;
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "  [dry-run] $*"
  else
    "$@"
  fi
}

write_file() {
  local path="$1"; shift
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "  [dry-run] write $path:"
    printf '            %s\n' "$@"
  else
    printf '%s\n' "$@" > "$path"
    echo "  wrote $path"
  fi
}

if [[ "$STATUS_ONLY" -eq 0 && "$DRY_RUN" -eq 0 && "$EUID" -ne 0 ]]; then
  echo "This script needs root to change system settings. Re-run with sudo." >&2
  exit 1
fi

echo "=============================================================="
echo " Raspberry Pi 5 tuning for Marine AI"
echo "   model : $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
echo "   kernel: $(uname -r)  ($(uname -m))"
echo "   cores : $(nproc --all)   RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "=============================================================="

# ======================================================================
# STATUS
# ======================================================================
print_status() {
  echo
  echo "--- current state ---"

  local gov="unknown"
  [[ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]] && \
    gov="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  echo "  CPU governor : $gov"
  [[ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq ]] && \
    echo "  CPU freq     : $(( $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) / 1000 )) MHz"

  if command -v vcgencmd >/dev/null 2>&1; then
    echo "  SoC temp     : $(vcgencmd measure_temp 2>/dev/null | cut -d= -f2)"
    local throttled
    throttled="$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2)"
    echo "  throttled    : $throttled $( [[ "$throttled" == "0x0" ]] && echo '(clean)' || echo '(see docs: bit0=under-voltage, bit1=freq-capped, bit2=throttled, bit3=temp-limit)' )"
  fi

  echo "  swap:"
  swapon --show --bytes 2>/dev/null | sed 's/^/    /' || echo "    (none)"
  echo "  zram module  : $(lsmod | grep -c '^zram' || true) instance(s) loaded"
  echo "  swappiness   : $(sysctl -n vm.swappiness)"
  echo "  jemalloc     : $(ls /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 2>/dev/null || echo 'NOT INSTALLED (sudo apt install libjemalloc2)')"

  echo "  USB autosuspend:"
  if [[ -r /sys/module/usbcore/parameters/autosuspend ]]; then
    echo "    usbcore.autosuspend = $(cat /sys/module/usbcore/parameters/autosuspend) (-1 disables it)"
  fi
  local suspended=0
  for f in /sys/bus/usb/devices/*/power/control; do
    [[ -r "$f" ]] || continue
    [[ "$(cat "$f")" == "auto" ]] && suspended=$((suspended + 1))
  done
  echo "    $suspended USB device(s) still set to 'auto' (power-managed)"

  echo "  video devices:"
  ls /dev/video* 2>/dev/null | sed 's/^/    /' || echo "    (none found)"
}

if [[ "$STATUS_ONLY" -eq 1 ]]; then
  print_status
  exit 0
fi

# ======================================================================
# 1. ZRAM
# ======================================================================
# Why ZRAM on a machine with 8 GB: the app itself needs ~1.5-2.5 GB, but the
# page cache is what makes model loads and image I/O fast, and a long field
# session gradually fills RAM with cold anonymous pages (Python arenas, one-shot
# imports, the ultralytics dataset cache). ZRAM compresses those to ~1/3 in RAM
# instead of evicting cache or swapping to the SD card -- an SD card swap on a
# Pi is slow enough to look like a hang, and it burns write cycles.
#
# zstd is the right algorithm on a Cortex-A76: ~2.8x compression at multi-GB/s,
# and the decompression cost is far below an SD card round trip.
echo
echo "--- 1/4  ZRAM ---"
if command -v apt-get >/dev/null 2>&1 && ! dpkg -s zram-tools >/dev/null 2>&1; then
  echo "  zram-tools is not installed."
  echo "  While the Pi still has internet: sudo apt install -y zram-tools"
fi

if [[ -d /etc/default ]]; then
  # Half of RAM as a compressed device is the usual recommendation; with ~2.8x
  # zstd compression that is ~4 GB of device backed by ~1.4 GB of real memory.
  write_file /etc/default/zramswap \
    "# Managed by Marine AI scripts/setup_pi_os.sh" \
    "ALGO=zstd" \
    "PERCENT=50" \
    "PRIORITY=100"
  if systemctl list-unit-files 2>/dev/null | grep -q '^zramswap.service'; then
    run systemctl enable zramswap.service
    run systemctl restart zramswap.service
    echo "  zramswap.service enabled"
  else
    echo "  NOTE: zramswap.service not present; install zram-tools to activate the config above."
  fi
fi

# Disable the SD-card swap file: once ZRAM exists there is no reason to ever
# touch the card, and dphys-swapfile competes with it for the same pages.
if systemctl list-unit-files 2>/dev/null | grep -q '^dphys-swapfile.service'; then
  run systemctl disable --now dphys-swapfile.service || true
  echo "  dphys-swapfile (SD-card swap) disabled"
fi

# With ZRAM, swapping is cheap, so let the kernel prefer it over dropping page
# cache. 100 is the standard ZRAM recommendation (default 60 assumes slow disk).
write_file /etc/sysctl.d/60-marine-ai-zram.conf \
  "# Managed by Marine AI scripts/setup_pi_os.sh" \
  "vm.swappiness=100" \
  "vm.page-cluster=0" \
  "vm.vfs_cache_pressure=50" \
  "vm.dirty_background_ratio=5" \
  "vm.dirty_ratio=15"
run sysctl --system >/dev/null
echo "  sysctl applied (swappiness=100, page-cluster=0 for ZRAM's 4K pages)"

# ======================================================================
# 2. CPU governor
# ======================================================================
# The Pi 5 defaults to 'ondemand', which ramps 1.5 -> 2.4 GHz only after it
# observes sustained load. A single 300 ms inference is over before the governor
# reacts, so every request effectively runs at the low clock. 'performance'
# pins 2.4 GHz and removes that ramp latency -- worth ~20-30% on bursty
# inference. It costs idle power and heat, which is why active cooling is a
# stated prerequisite for this deployment.
echo
echo "--- 2/4  CPU governor -> performance ---"
if [[ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]] || [[ "$DRY_RUN" -eq 1 ]]; then
  if command -v apt-get >/dev/null 2>&1 && ! dpkg -s cpufrequtils >/dev/null 2>&1; then
    echo "  NOTE: install cpufrequtils to make this persist across reboots:"
    echo "        sudo apt install -y cpufrequtils"
  fi
  write_file /etc/default/cpufrequtils \
    "# Managed by Marine AI scripts/setup_pi_os.sh" \
    "GOVERNOR=\"performance\""

  if [[ "$DRY_RUN" -eq 0 ]]; then
    for cpu in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
      echo performance > "$cpu" 2>/dev/null || true
    done
    echo "  governor set on all cores: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  fi

  # A systemd unit is the reliable way to reapply it at boot; cpufrequtils alone
  # sometimes runs before all cores are online.
  write_file /etc/systemd/system/cpu-performance.service \
    "[Unit]" \
    "Description=Pin CPU governor to performance (Marine AI)" \
    "After=multi-user.target" \
    "" \
    "[Service]" \
    "Type=oneshot" \
    "RemainAfterExit=yes" \
    "ExecStart=/bin/sh -c 'for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do echo performance > \$c; done'" \
    "" \
    "[Install]" \
    "WantedBy=multi-user.target"
  run systemctl daemon-reload
  run systemctl enable cpu-performance.service
  echo "  cpu-performance.service enabled (reapplies on boot)"
else
  echo "  SKIP: cpufreq sysfs not writable (not a Pi, or a container)"
fi

# ======================================================================
# 3. USB power management
# ======================================================================
# UVC microscopes are the classic victim of USB autosuspend: the kernel powers
# the device down between captures, and the next frame grab either stalls for
# hundreds of ms while it wakes or returns -EIO and drops the stream. Turning
# autosuspend off costs a fraction of a watt and removes a whole class of
# intermittent "camera disconnected" bugs in the field.
echo
echo "--- 3/4  USB power management ---"
write_file /etc/modprobe.d/marine-ai-usb.conf \
  "# Managed by Marine AI scripts/setup_pi_os.sh" \
  "# -1 disables USB autosuspend globally (keeps the microscope awake)." \
  "options usbcore autosuspend=-1"

# modprobe.d only applies at module load, i.e. next boot. Apply it live too.
if [[ "$DRY_RUN" -eq 0 ]]; then
  [[ -w /sys/module/usbcore/parameters/autosuspend ]] && \
    echo -1 > /sys/module/usbcore/parameters/autosuspend 2>/dev/null || true
  changed=0
  for f in /sys/bus/usb/devices/*/power/control; do
    [[ -w "$f" ]] || continue
    echo on > "$f" 2>/dev/null && changed=$((changed + 1))
  done
  echo "  autosuspend disabled on $changed USB device(s) (live)"
fi

# A udev rule keeps devices plugged in later (a microscope swapped mid-cruise)
# from silently reverting to power-managed.
write_file /etc/udev/rules.d/99-marine-ai-usb.rules \
  "# Managed by Marine AI scripts/setup_pi_os.sh" \
  "# Keep every USB device powered on; UVC microscopes drop frames when suspended." \
  'ACTION=="add", SUBSYSTEM=="usb", TEST=="power/control", ATTR{power/control}="on"' \
  "# Give video devices a stable name regardless of enumeration order." \
  'SUBSYSTEM=="video4linux", SUBSYSTEMS=="usb", ATTRS{bInterfaceNumber}=="00", SYMLINK+="microscope"'
run udevadm control --reload-rules
echo "  udev rules reloaded (/dev/microscope symlink available after replug)"

# The Pi 5's USB ports can deliver 1.6 A total, but only when the firmware knows
# a 5 A supply is attached. Without this, a bus-powered microscope plus storage
# can brown out under load.
echo
echo "  Firmware note: for the full 1.6 A USB budget, ensure /boot/firmware/config.txt has:"
echo "      usb_max_current_enable=1"
if [[ -f /boot/firmware/config.txt ]] && ! grep -q '^usb_max_current_enable=1' /boot/firmware/config.txt; then
  if [[ "$DRY_RUN" -eq 0 ]]; then
    printf '\n# Added by Marine AI scripts/setup_pi_os.sh\nusb_max_current_enable=1\n' >> /boot/firmware/config.txt
    echo "  appended usb_max_current_enable=1 to /boot/firmware/config.txt (reboot required)"
  else
    echo "  [dry-run] would append usb_max_current_enable=1 to /boot/firmware/config.txt"
  fi
fi

# ======================================================================
# 4. Misc
# ======================================================================
echo
echo "--- 4/4  misc ---"
# The service user must be able to open /dev/video*.
TARGET_USER="${SUDO_USER:-pi}"
if id "$TARGET_USER" >/dev/null 2>&1; then
  if ! id -nG "$TARGET_USER" | tr ' ' '\n' | grep -qx video; then
    run usermod -aG video "$TARGET_USER"
    echo "  added $TARGET_USER to the 'video' group (log out/in to take effect)"
  else
    echo "  $TARGET_USER is already in the 'video' group"
  fi
fi

if command -v apt-get >/dev/null 2>&1; then
  MISSING=()
  for pkg in libjemalloc2 v4l-utils zram-tools cpufrequtils; do
    dpkg -s "$pkg" >/dev/null 2>&1 || MISSING+=("$pkg")
  done
  if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "  Install these while the Pi still has internet:"
    echo "      sudo apt install -y ${MISSING[*]}"
  fi
fi

print_status

cat <<'EOF'

==============================================================
 Reboot to apply the boot-time settings (ZRAM, governor unit,
 usbcore autosuspend, USB current limit):

     sudo reboot

 Then confirm with:
     ./scripts/setup_pi_os.sh --status
     python scripts/pi_healthcheck.py
==============================================================
EOF
