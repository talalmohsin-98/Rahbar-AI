# Screenshots

The images referenced by the root [README.md](../../README.md) live here. They
were captured from the app running locally at 1440 CSS px wide, 2× device
pixel ratio. `assistant-pipeline.png` uses a taller viewport so the whole
inspector panel fits in one frame.

| Filename | Page / State |
|---|---|
| `landing.png` | Home (`/`) — hero, search box, services preview, how-it-works, FAQ |
| `services.png` | Services grid (`/services`) — all 5 department cards |
| `recommend.png` | Get Help wizard (`/recommend`) — step 3 result card (docs / fee / time) |
| `assistant.png` | Assistant (`/assistant`) — a cited answer with its verification badges |
| `assistant-pipeline.png` | Assistant (`/assistant`) — same answer with the Pipeline Inspector open |
| `about.png` | About (`/about`) — methodology + tech stack |

## Regenerating them

Start the backend (`uvicorn main:app --port 8000` in `backend/`) and the
frontend (`npm run dev` in `frontend/`), then drive a browser through the six
states above and save each capture under the filename in the table. The root
README references these paths directly, so a replacement file with the same
name is picked up with no README edit.
