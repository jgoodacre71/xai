---
name: notebook-polisher
description: Improve a demo notebook's structure and clarity. Use when a notebook works technically but needs clearer markdown, tighter sequencing, cleaner headings, stronger figure ordering, or more explicit takeaways and caveats.
---

# Notebook Polisher

## Purpose

Make notebooks teach, not merely execute.

## Checklist

- Open with why the demo matters.
- State the task and dataset clearly.
- Introduce the model and explanation methods briefly.
- Make figures appear near the text that interprets them.
- Add a crisp "What we learned" section.
- Add a "Residual risks" or "What this does not prove" section.
- Keep code cells focused and avoid large hidden helper blocks.
- Move reusable helper code into `src/` if it is not already there.

## Behavioural XAI demo guidance

For Clever-Hans or shortcut notebooks, make the core evidence behavioural:

- stage apparent success before revealing the shortcut;
- show same-object counterfactuals with predicted label, model score, and a
  visible "what changed / what stayed fixed" strip;
- keep saliency as supporting evidence, not the main claim;
- add a re-test after intervention, using the same hard cases;
- fail the notebook if the intended behavioural story is weak or uncertain;
- when scoring generated diagnostic frames, score the image tensor directly
  rather than routing temporary sample objects through identity-based caches.
