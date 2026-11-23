-- 002_seed_v1.sql: register the notebook model as version 1.
-- model_key is null because v1 is the .h5 bundled in models/.
-- Metrics are the full test set results from models/metrics.json.

insert into training_runs (status, promoted, model_version, model_key, challenger_metrics, started_at, finished_at)
select 'done', true, 1, null,
       '{"accuracy": 0.9092578986039677, "loss": 0.366781021072749, "precision_macro": 0.8841960838903342, "recall_macro": 0.8943644396887744, "f1_macro": 0.8857537363230963, "precision_weighted": 0.9129088961028744, "recall_weighted": 0.9092578986039677, "f1_weighted": 0.908875664089224, "top3_accuracy": 0.9893460690668626, "roc_auc_ovr_macro": 0.9939001658139984}'::jsonb,
       '2026-11-22T21:23:50Z', '2026-11-22T21:23:50Z'
where not exists (select 1 from training_runs where model_version = 1);
