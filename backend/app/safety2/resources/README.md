# Safety resource store (Book Vol 6 Ch 7)

`global.yaml` holds broadly-applicable international resources. `regions/<region_code>.yaml`
holds resources specific to a region code (matches `UserProfile.region`, e.g. `us`, `uk`) —
loaded and layered on top of `global.yaml` by `app.safety2.worker.load_resources(region)`.

## Status

Populated with publicly published, well-known crisis-support services (see each file's
`source_notes`). This is a snapshot as of each file's `last_updated` date, not a standing
verification process — `app.safety2.worker.resources_are_stale()` flags a file whose
`last_updated` is older than its `stale_after_days` (default 180) so an unreviewed store is
surfaced rather than silently kept serving.

**NEEDS REVIEW by a qualified safety/clinical professional on an ongoing basis** before
real-user exposure, and periodically thereafter — same caveat as the skills library
(`app/skills/*/manifest.yaml`'s `source:` fields).

## Adding a region

Add `regions/<region_code>.yaml` with the same shape as `us.yaml`/`uk.yaml`:
`last_updated`, `source_notes`, and a `resources` list of
`{id, title, description, contact, region, category, verified}` entries. `region_code` should
match the values you intend to store in `UserProfile.region`.
