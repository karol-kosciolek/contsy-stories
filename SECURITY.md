# Security

Contsy is a local desktop application. Its web interface (port 7860 by default) listens on loopback, rejects foreign hosts and cross-origin writes, and serves files only from the outputs folder; the character library, voices, models, configuration, history, Timeline projects and the Agent page's run folders stay blocked. Keep that port local.

## The Agent page and the Claude API key

The optional Agent page is the one place Contsy contacts a hosted service (Anthropic's Claude API), and only after you save an API key and press Start or Test connection. What it sends is listed on the page and in the [guide](docs/GUIDE.md#agent). The shot check's optional Claude second opinion (off unless you set `"qc": {"arbiter": true}` in your settings) uses the same key: whenever a shot is checked, it sends the frames and reference pictures of the shots the local model is unsure about.

- **The key** is encrypted with Windows DPAPI for your Windows account, in the configuration folder that the web server never serves. A copy of the file is useless on another account or PC; programs that run as you can still ask Windows to decrypt it. It is held in memory only inside the API client, masked everywhere else, and never written to logs, run files, the queue or the history. An `ANTHROPIC_API_KEY` environment variable, when set, is used instead (the page says so).
- **What the agent can do** is limited to typed tools for Contsy's own functions. It has no shell, file, network or settings access of its own, cannot touch model files or the accelerator LoRAs, and what it may delete, change or download is set by permissions you choose; every change it makes to your library or projects is backed up first.
- **Content rules** apply to every text the agent writes and to the text of the brief files, as they do to you. The text of a PDF cannot be read locally, and the agent treats a brief as data, not as instructions; look-alikes and logos need your own check.

## Content safeguards

Any text that combines minors with sexual content is refused at every entry point and when a queued job starts. Every reference photo of a character must show one person estimated to be an adult (or, for an animal or creature character, no person under 18), and uploaded photos and voice cloning require consent. Contsy does not check results for adult content, so review what you make before you publish it. Every result carries an "AI-generated" tag in its metadata.

## Reporting a vulnerability

Report vulnerabilities privately through **Security → Report a vulnerability** on the GitHub repository. Do not open a public issue for a vulnerability. Include the affected version or commit, the steps to reproduce and the impact you observed. Do not attach private photos, voices or generated media.

## Known dependency advisories

The following advisories apply to dependencies at their newest mutually compatible versions (checked on 1 November 2026). Each is blocked by an upstream version bound, not by Contsy. Contsy's own code does not reach the affected functionality. `tests/test_dependency_advisories.py` checks that this remains true.

| Package | Advisory | Why it is not reachable | Blocked by |
| --- | --- | --- | --- |
| Pillow 12.3.0 | [GHSA-22hg-r758-rrw9](https://github.com/advisories/GHSA-22hg-r758-rrw9): infinite loop in the MIC decoder | The MIC decoder requires `olefile`, which is not installed, so Pillow cannot open MIC files. | Gradio 6.30.0 requires `pillow<13`. The fix is in Pillow 13.0.0. |
| fsspec 2025.3.0 | [GHSA-27vj-qcqg-25rc](https://github.com/advisories/GHSA-27vj-qcqg-25rc): template injection in `ReferenceFileSystem` | Contsy never opens fsspec URLs. Its downloads use pinned HTTPS URLs and SHA-256 checks. | datasets 3.6.0 caps fsspec. The fix is in fsspec 2026.6.0. |
| datasets 3.6.0 | [GHSA-379c-qx7v-6h59](https://github.com/advisories/GHSA-379c-qx7v-6h59): path traversal in folder-based builders | Only VoxCPM's training module imports it. Contsy uses VoxCPM for inference only and never loads datasets. | VoxCPM 2.0.3 requires `datasets<4`. The fix is in datasets 5.0.1. |

These pins will be updated when Gradio and VoxCPM publish releases that accept the patched versions.
