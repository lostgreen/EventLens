# Paper Figures

These PNG illustrations are rendered from the EventLens paper,
*Event-Structure Reinforcement Learning for Video Understanding*.

| Preview | Original Figure | Purpose |
| --- | --- | --- |
| `overview.png` | `figures/Figre1_new.pdf` | Event boundary, progression, and relation recovery |
| `training-pipeline.png` | `figures/Figure2.pdf` | Specialist GRPO followed by MT-OPD |
| `event-structure.png` | `figures/eventlens_annotation_core.pdf` | Scene-first annotation and temporal hierarchy |
| `data-domains.png` | `figures/input_domain_sunburst.pdf` | Source-video domains before recovery-task construction |

Source snapshot: `095c319a6018854bf1ddcac77ec5632915291b33` in the paper
repository. All source PDFs were unchanged relative to that snapshot.
The event-structure illustration uses report frames; its boundaries and event
assignments are schematic, as stated in the paper caption.

Paper illustrations use the complete figure page rendered by Poppler at a
maximum dimension of 2,000 pixels (1,600 for the domain diagram). Their content
is unchanged. The domain diagram describes source videos, not task-mix counts.

`provenance.json` records source and output hashes. Manuscript PDFs and
editable paper authoring files remain in the paper repository.

The README result table reproduces the pretrained and EventLens 4B/8B rows from
`sections/experiments.tex` (Tables `tab:temporal` and `tab:general`) at the
same paper snapshot for five selected benchmarks. It contains original scores,
no averaging or relative-percentage conversion, and best scores marked within
each base-model/EventLens pair.

These are paper illustrations, not training data. The code license does not
grant separate rights to depicted source-video content. Source-specific
conditions are described in the data documentation.
