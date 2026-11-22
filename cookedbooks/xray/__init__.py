"""X-ray: a read-only, jury-side "case file" for one injected error.

`python -m cookedbooks.xray EPISODE_DIR [--issue ID | --all] [--color auto|always|never]`

It reads the whole episode folder, including the closed files `truth.json`
and `provenance.json`, and never writes anything. It is a display tool for a
human (a reviewer or a demo); no environment, policy or grader module imports
it, and it imports only the standard library, `cookedbooks.episode.loader`
and `cookedbooks.grader.truth`.
"""
