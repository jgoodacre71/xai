# Spec TODO

This file tracks the major remaining gaps against `REPO_SPEC.md`. It is the
durable todo list for work that should not live only in chat history.

## Completed flagship work

- [x] Foundation package, tests, CLI entry points, and anti-drift task memory
- [x] Real MVTec AD PatchCore hero demo with nearest-normal provenance
- [x] PatchCore wrong-normal demo
- [x] PatchCore limits demos for count, severity, and logic
- [x] Real MVTec LOCO AD path for Demo 07
- [x] Explicit pretrained ResNet-18 feature-map PatchCore path
- [x] Demo-ready suite runner and static presentation index
- [x] Demo 03 local benchmark diagnostics over the real MVTec bottle test split
- [x] Dataset governance docs for MVTec AD and MVTec LOCO AD, including
  non-commercial restrictions

## Highest-priority remaining gaps

- [x] Add a real Waterbirds or equivalent shortcut dataset adapter and local
  preparation workflow
- [x] Replace Demo 01's synthetic proxy with a real classifier-based shortcut
  demo using worst-group metrics
- [x] Add saliency and perturbation evidence for Demo 01, such as Grad-CAM and
  Integrated Gradients
- [x] Add a stronger real industrial shortcut dataset path for Demo 02 or a
  neural industrial shortcut baseline
- [ ] Add a component-aware or logic-aware comparator for Demo 07
- [x] Extend Demo 08 from synthetic shift to a real corruption or acquisition
  shift path

## Strong additions from the spec

- [ ] Add MVTec AD 2 support
- [ ] Add VisA support
- [ ] Add MetaShift or Spawrious support
- [ ] Consider ProtoPNet or another interpretable comparator where it improves a
  demo rather than bloating the repo

## Notes

- This repo is intended for academic / non-commercial work. The MVTec family is
  documented as non-commercial in `docs/DATASETS.md`.
- When a todo is completed, the corresponding task should also be recorded under
  `docs/tasks/completed/`.
