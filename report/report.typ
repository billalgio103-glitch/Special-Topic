// Built by `just` (src/build_report.py). Every number and figure is read from the
// pipeline outputs, so new training data changes this report on the next run.
#let data = sys.inputs.at("data", default: "/data")
#let r = json(data + "/model/results.json")
#let v = json(data + "/validation.json")
#let t = r.test
#let f(x) = if x == none { "–" } else { str(calc.round(x, digits: 3)) }

#set document(title: "Nanoparticle segmentation: model report (Part I)")
#set page(margin: 2cm, numbering: "1")
#set text(size: 10pt)
#set heading(numbering: "1.")

#align(center)[
  #text(16pt, weight: "bold")[Nanoparticle segmentation in electron micrographs]\
  Model & Report Part I · EMA 5937 · Junyoung
]

= Summary
A small U-Net segments particles in 256 × 256 electron-microscopy images. On the
held-out test images it reaches a pixel ROC AUC of *#f(t.unet.auc)*, against
#f(t.dummy.auc) for the negative control and #f(t.positive_control.auc) for the
positive control (expected #f(r.controls.positive_expected_auc)). The model is not
tuned yet; Part II will improve it.

= Data
#v.source_counts.values().sum() images from three public sources (EMPS, HRTEM, Co3O4) were
built into one SQLite database (fingerprint #raw(v.database_fingerprint.slice(0, 12))…).
Splits are by group (source paper or acquisition folder), so related images never
cross between training and testing.

#table(
  columns: 4, align: (left, right, right, right),
  [*Partition*], [*Images*], [*Use*], [*File*],
  [fit], [#r.split_sizes.fit], [gradient updates], [`train.safetensors`, fold ≠ #r.config.val_fold],
  [validation], [#r.split_sizes.validation], [epoch selection], [`train.safetensors`, fold = #r.config.val_fold],
  [test], [#r.split_sizes.test], [final metric only], [`eval.safetensors`],
)

#figure(image(data + "/figures/dataset_overview.png", width: 85%), caption: [Images and foreground fraction per source and split.])

= Model and training
U-Net with three levels (width #r.config.width), loss = #r.config.loss, Adam
(learning rate #r.config.lr), batch #r.config.batch_size, #r.config.epochs epochs, seed
#r.config.seed, torch #r.config.torch on CPU. The epoch with the best validation AUC
(epoch #r.best_epoch) is kept.

#figure(image(data + "/model/figures/training_curve.png", width: 100%), caption: [Training and validation loss, and validation pixel AUC per epoch.])

= Held-out test results

#table(
  columns: 5, align: (left, right, right, right, right),
  [*Model*], [*Pixel AUC*], [*EMPS*], [*HRTEM*], [*Co3O4*],
  ..for (key, label) in (("unet", "U-Net"), ("dummy", "Negative control"), ("positive_control", "Positive control")) {
    let m = t.at(key)
    let s = m.per_source_auc
    (label, f(m.auc), f(s.at("emps", default: none)), f(s.at("hrtem", default: none)), f(s.at("cobalt", default: none)))
  },
)
U-Net Dice at threshold 0.5: #f(t.unet.at("dice_at_0.5")). Co3O4 has no test
images: its patches have no known parent image, so they all stay in training.

*Controls.* Negative: #r.controls.negative. Positive: #r.controls.positive. Both
must hit their expected AUC or the pipeline stops, which checks the metric code
itself.

#figure(image(data + "/model/figures/roc.png", width: 55%), caption: [ROC curves on held-out test pixels.])

#figure(image(data + "/model/figures/examples.png", width: 75%), caption: [Held-out test examples: input, true mask, predicted probability.])

= Reproducibility
Inputs: `train.safetensors` #raw(r.inputs.train.sha256.slice(0, 12))…,
`eval.safetensors` #raw(r.inputs.eval.sha256.slice(0, 12))…. `make` retrains only
when these files or the model code change, then redraws the figures and this PDF.
