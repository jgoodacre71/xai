# IEEE DataPort scouting

IEEE DataPort should be used as a controlled candidate register, not as a
random data hunt. No IEEE dataset is selected for the core demos until access,
licence, work permission, and citation requirements are recorded.

## Candidate schema

Each candidate should record:

- `slug`
- `title`
- `ieee_dataport_url`
- `doi`
- `access_type`: `open_access`, `standard`, `competition`, or `unknown`
- `licence`
- `citation`
- `domain`
- `task`
- `data_type`: `image`, `time_series`, `tabular`, `sensor`, or `other`
- `size`
- `download_status`
- `work_permission_status`
- `suitable_demos`
- `why_interesting`
- `risks`
- `next_action`

## Fit categories

| Fit | Meaning |
| --- | --- |
| A | Excellent candidate; improves a core demo |
| B | Useful backup |
| C | Interesting but not urgent |
| D | Poor fit or unclear terms |

## Candidate roles

Look for datasets that can strengthen one of these roles:

- industrial defect images for marker shortcut auditing;
- industrial images for PatchCore anomaly provenance;
- sensor or time-series data for explanation drift;
- power or fault datasets for channel/window attribution;
- thermal or electrical inspection data for anomaly localisation.

## Permission notes

IEEE DataPort includes standard datasets, open-access datasets, and competition
datasets. Standard datasets may require subscriber access. IEEE DataPort
datasets require attribution/citation. Record access and citation details
before downloading or using any candidate at work.
