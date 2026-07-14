# Windows setup — what it actually takes

A start-to-finish record of getting PrusaPATuner running on a Windows 11 machine
(July 2026), including every place the README's install section was not enough and
what fixed it. If the README steps fail for you, this is the document to read.

## TL;DR — the working sequence

```powershell
# from the repo root, in a regular (non-admin) PowerShell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .

# run it (activation of the venv is optional — calling the venv exe directly works)
.\.venv\Scripts\prusa-pa-tuner.exe
```

Then do the **firewall check** below — on this machine the app came up looking
perfectly healthy while Windows was set to drop every printer packet.

> **`No module named prusa_pa_tuner`?** You ran `python -m prusa_pa_tuner` with
> the *global* Python — the package only exists inside `.venv`. Either activate
> the venv first (`.\.venv\Scripts\activate`) so plain `python` resolves to it,
> or call the venv directly: `.\.venv\Scripts\python.exe -m prusa_pa_tuner` or
> `.\.venv\Scripts\prusa-pa-tuner.exe`. Activation lasts only for the current
> shell session — a new terminal needs it again.

## Prerequisites that matter

- **Python 3.11+ from python.org**, not the Microsoft Store. This machine had
  Python 3.12.10 at `C:\Users\<you>\AppData\Local\Programs\Python\Python312\`.
  Store Python works but re-triggers the firewall trap after every auto-update
  (see the README's "Windows trap" section); python.org Python has a stable path,
  so the firewall fix below sticks.
- Check what you have with `py --list` and `(Get-Command python).Source`.

## Failure 1 — `pip install -e .` refused to build (fixed in the repo)

With current setuptools (77+, pulled in automatically as the build backend), the
install failed twice on `pyproject.toml` metadata that was fine when the project
started:

1. `` configuration error: `project.license` must be string `` — the old
   `license = { text = "AGPL-3.0-or-later" }` table form is rejected once
   `license-files` is also set. Fixed to the PEP 639 SPDX string form:
   `license = "AGPL-3.0-or-later"`.
2. `InvalidConfigError: License classifiers have been superseded by license
   expressions` — the `License :: OSI Approved :: …` classifier is a hard error
   when an SPDX expression is present. The classifier was removed.

Both fixes are committed, so a fresh clone should not hit this — but if you see
either message, this is why, and pinning old setuptools is *not* the right fix.

## Failure 2 — never run `pip install` while the app is running

`pip install -e ".[dev]"` while `prusa-pa-tuner.exe` was serving failed with
`WinError 32: The process cannot access the file … prusa-pa-tuner.exe` — Windows
locks running executables. Worse, pip had already **removed the old install
before failing**, leaving the venv broken (`ModuleNotFoundError:
prusa_pa_tuner`) and a corrupt `~rusa-pa-tuner` remnant in `site-packages`.

Recovery:

```powershell
# stop the app first, then:
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Get-ChildItem .venv\Lib\site-packages -Filter '~rusa*' -Force |
  Remove-Item -Recurse -Force            # clean the half-uninstalled remnant
.\.venv\Scripts\python.exe -m pip check   # should say "No broken requirements"
```

Rule of thumb: **stop the server before any `pip install` in this venv.**

## Failure 3 — the firewall silently drops printer packets

This is the big one, because *nothing looks wrong*: the app starts, the UI loads,
the UDP listener binds `0.0.0.0:8514`, tests pass — and every packet from the
printer dies at the Windows firewall.

What was found on this machine:

- **Two enabled inbound `Block` rules for
  `…\Programs\Python\Python312\python.exe`** (created whenever a firewall
  prompt for Python was dismissed in the past), scoped to the **Public** profile.
- **Both active networks were classified Public** (the WLAN and the NordLynx VPN
  adapter), so the Block rules applied.
- The venv's `python.exe` is only a launcher — the process that actually binds
  the UDP socket is the **base** `Python312\python.exe`, so those Block rules
  govern the venv app too. A port-based "allow UDP 8514" rule does **not** help;
  the per-program Block rule wins.

Diagnose (regular PowerShell):

```powershell
Get-NetFirewallApplicationFilter | Where-Object { $_.Program -like '*python*' } |
  ForEach-Object { $r = $_ | Get-NetFirewallRule;
    "{0} | enabled={1} action={2} dir={3} profile={4} prog={5}" -f
      $r.DisplayName, $r.Enabled, $r.Action, $r.Direction, $r.Profile, $_.Program }

Get-NetConnectionProfile | Select-Object Name, InterfaceAlias, NetworkCategory
```

Any `enabled=True action=Block` line pointing at your Python, on a profile your
LAN uses, is fatal. Fix (**elevated** PowerShell — this cannot be automated from
a normal shell):

```powershell
Get-NetFirewallApplicationFilter |
  Where-Object { $_.Program -like '*python312*python.exe' } |
  Get-NetFirewallRule | Where-Object { $_.Action -eq 'Block' } |
  Set-NetFirewallRule -Action Allow
```

(Adjust the `*python312*` pattern to your Python path.) Alternatively: Windows
Security → Firewall & network protection → *Allow an app through firewall* →
tick both boxes for the Python entries.

*Applied on this machine 2026-07-14: both rules flipped to `Allow`, and printer
UDP data confirmed flowing immediately afterwards.*

Then verify end-to-end with the printer on and metrics configured (Step 0 of the
README):

```powershell
.\.venv\Scripts\python.exe scripts\sniff.py --prusalink-host <printer-IP> --password <PrusaLink-password>
```

`loadcell_value` ticking at ~180 Hz = you're done. Silence = re-read this section.

### VPN caveat

A NordVPN (NordLynx) adapter was active on this machine. If packets still don't
arrive after the firewall fix, check the VPN client's LAN/invisibility settings —
some VPN clients block inbound LAN traffic independently of the Windows firewall,
and the printer must be able to reach this host's **LAN** address, not the VPN
address. The app prints the IP it detected toward the printer on startup; make
sure the printer's Metrics & Log host matches it.

## Running and verifying

```powershell
.\.venv\Scripts\prusa-pa-tuner.exe                # opens browser at http://127.0.0.1:8765/
.\.venv\Scripts\prusa-pa-tuner.exe --no-browser   # headless
```

On startup it prints the config path (`%APPDATA%\PrusaPATuner\config.json`), the
HTTP URL, and the UDP port. Health checks that all passed on this machine:

- `Invoke-WebRequest http://127.0.0.1:8765/` → HTTP 200
- `Get-NetUDPEndpoint -LocalPort 8514` → bound to `0.0.0.0`
- `.\.venv\Scripts\python.exe -m pytest -q` → 126 passed (needs the `[dev]` extra)

## Known-good versions (this machine, 2026-07-14)

Python 3.12.10 · pip 25.x · setuptools 80.x (build-time) · fastapi 0.139.0 ·
uvicorn 0.51.0 · numpy 2.5.1 · scipy 1.18.0 · pydantic 2.13.4 · httpx 0.28.1
