> Runtime data now ships with this course module as `components/ml_librarian/deployment/runtime.zip`. First open automatically installs and verifies it. No `RUNTIME_BUNDLE_URL` or separate data hosting is required. Update component exports with `--include-runtime`; Groq setup is only for generated answers.

# ML Librarian integrated into the course

Explicit user authorization on2026-10-18 permits only deployment-related course
changes; other course content remains out of scope. This is a module of the
existing course app, not an iframe, redirect or separately hosted application.

Navigation group36contains:
- `/ml_librarian`: evidence search and optional server-configured answer.
- `/ml_librarian_lecturer`: current instructor documentation and notebook downloads.

Entrypoint stays `app.py`, Python stays3.11. The host owns page configuration and
navigation. Thin files views/07_ml_librarian*.py call components/ml_librarian/
embedded.py. No neural initialization happens just by visiting the course home
or lecturer page. Only opening search prepares/loads pinned models. No training,
index construction or source-course execution happens within the module.

The component is a versioned source export from Kilitar/ML_Librarian, with
export_manifest.json binding source commit/worktree status and SHA per file.
Do not hand-edit exported files: edit source repository and run its
src/librarian/export_course_module.py --course-root PATH_TO_COURSE.

## Existing Streamlit course deployment

Publish this course commit to its existing remote branch and rebuild the
EXISTING Streamlit app. Keep its current app.py entrypoint; do not deploy the
component's development cloud_app.py as another application. requirements.txt
now includes component dependencies and needs a dependency rebuild.

Configure the EXISTING app's server Secrets using .streamlit/secrets.example.toml:
RUNTIME_BUNDLE_URL, RUNTIME_BUNDLE_SHA256, GROQ_API_KEY and GROQ_MODEL. Runtime
artifact/SHA come from component deployment/artifacts.json; host the course data
at an authorized HTTPS endpoint. Models download pinned public revisions once;
query-time inference is offline. Lecturer supplies no API key. Use a dedicated
Groq Free Plan account; the application cannot verify billing tier itself.

Missing data/key/model/limit errors are scoped to Librarian; other pages remain
available. The minimum runtime bundle includes extracted text, not original
course files to download. Local runtime data and model caches/secrets are ignored
by Git. No immutable retrieval snapshot was regenerated for this integration.

Optional local environment ML_LIBRARIAN_CACHE points to an existing compatible
cache. It is not required on Cloud and must not be a dependency on a sibling repo.

## Verification scope

Navigation AST validates all221pagefiles. Librarian129unit tests passed. Actual
course navigation/search parity and local HTTP are recorded in the source repo
handoff. Linux/Python3.11/cloud memory/hosted URL and live Groq answer quality
remain unverified until the existing deployment rebuilds with secrets. Current
lecturer layout is integrated unchanged; redesign is a later user review task.


## Right-hand dock

The course common entrypoint renders a floating Librarian launcher at bottom-right before nav.run(). Open it without leaving the lesson. Submitted query, filters, results and answer persist in session memory across closing/navigation. Closed dock does not initialize search; dock/full-page share one cached index. Instructor documentation remains separate. See components/ml_librarian/docs/RIGHT_DOCK.md for validation and styling limits. Desktop/mobile CSS still requires visual review.
