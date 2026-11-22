# Documentation

This directory contains the English documentation for the project. It covers the data specification, governance and modeling reports, the dashboard, deployment, and figure reproduction. The Chinese originals live in the parent [`docs/`](../README.md) directory.

## Data and Scope

| Document | Content |
|---|---|
| [data_source](data_source.md) | Dataset, raw files, checksums, citation, and usage terms |
| [data_dictionary](data_dictionary.md) | Raw fields, missing rates, and cleaning rules |
| [feature_dictionary](feature_dictionary.md) | 47-column wide table and 36 model features |
| [scope_definition](scope_definition.md) | Game-product boundary and `in_scope` rules |
| [scope_audit](scope_audit.md) | Boundary candidates and Retro review list |

## Methods and Results

| Document | Content |
|---|---|
| [governance_report](governance_report.md) | Missingness mechanisms, imputation, outliers, and consistency checks |
| [classification_report](classification_report.md) | Classifiers, 5/10-fold evaluation, calibration, error analysis, and SHAP |
| [rules_review](rules_review.md) | Association rules, threshold sensitivity, same-series and redundancy checks |

## Running and Presentation

| Document | Content |
|---|---|
| [setup](setup.md) | Environment, dependencies, fonts, and troubleshooting |
| [figure_reproduction](figure_reproduction.md) | Rebuilding 44 figures, input dependencies, and validation |
| [dashboard_guide](dashboard_guide.md) | Streamlit dashboard, linked filters, and public entries |
| [streamlit_deploy](streamlit_deploy.md) | Deployment on Streamlit Community Cloud |

## Language Policy

The English files are full translations of the Chinese sources, not summaries. Numbers, tables, commands, code identifiers, and file paths are kept identical. If a wording difference appears, the Chinese source is the authoritative reference; please open an issue or pull request so the English version can be corrected.

Chinese originals: [docs/README.md](../README.md). Project overview: [README.md](../../README.md).
