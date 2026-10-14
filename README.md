# EventLens Project Page

Academic project website for **EventLens: Event-Structure Reinforcement Learning for Video Understanding**.

Live page: [lostgreen.github.io/EventLens](https://lostgreen.github.io/EventLens/).

Paper: [arXiv:2610.16172](https://arxiv.org/abs/2610.16172).

Dataset: [EventLens-Train](https://huggingface.co/datasets/Lostgreen/EventLens-Train).

Student models: [EventLens-4B](https://huggingface.co/Lostgreen/EventLens-4B) · [EventLens-8B](https://huggingface.co/Lostgreen/EventLens-8B).

This repository contains the project website. The training code link will be added when its public release is available.

## Edit and preview

Edit `index.html` for content and `static/css/index.css` for styling. Paper illustrations live in `static/images/`; complete original-paper table crops (Tables 1, 2, 4, and 6) live in `static/tables/`.

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000`. No build or package installation is required.

GitHub Pages uses the `main` branch and repository root. Keep `.nojekyll` in the root.

## Credits

Adapted from the [Academic Project Page Template, original-version](https://github.com/eliahuhorwitz/Academic-project-page-template/tree/original-version) (commit `b2e6e09010f0feb72b115fe9cc6bcb9d1277047c`) and [Nerfies](https://nerfies.github.io/).

The adapted website layout is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Bulma is MIT-licensed; see `THIRD_PARTY_NOTICES.md`. Research illustrations and depicted third-party media retain their original rights. This website license does not license the datasets or model weights.
