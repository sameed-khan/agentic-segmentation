# Phase 4 datasets — verified facts and split design

Phase 4 extends the MSTT-199 soft-tissue-tumour harness to **general 3D lesion segmentation**.
All facts below were verified from primary sources on 2026-10-19 (journal landing pages,
figshare API, DataCite API, TCIA collection pages, Zenodo/DataCite records). Leakage defence
note: our models' training cutoffs are around mid-2026, so for the *additional* datasets we
verified the **first public data release** date, not just the paper.

## Verified facts table

| # | Dataset | Body region / lesion | Modality | Cases | Lesions | Paper (DOI, pub date) | Data release (DOI, date) | Licence | Size | Baseline | Verified release notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | BEAMSTER | brain, metastases | CE-T1 MRI (registered to planning-CT space, defaced) | 140 (A=113, B=27 small-lesion-enriched) | 260 (binary masks, multiple lesions per case) | Sci Data 2026-07-01, 10.1038/s41597-026-07777-0 (received 2025-10-17, accepted 2026-06-26) | figshare 10.6084/m9.figshare.29365844, **2026-06-01** | CC BY 4.0 | 4.34 GB, 285 files | none published in the accessible version (full text truncated on nature.com; no baseline Dice verified) | Data release precedes paper by 1 month; no earlier preprint found (Europe PMC/S2/arXiv) |
| 2 | Spinal-Multiple-Myeloma-SEG | spine, multiple-myeloma focal lesions (+ vertebra masks w/ type labels) | dual-energy low-dose CT (Philips IQon; conv. + VMI 40/80/120 keV + Ca-suppressed SI 25/50/75/100) | 67 patients / 72 scans / 576 series / 564,464 slices | multiple focal lesions per scan; lesion masks (binary; vertebra masks carry class labels) | Sci Data 2026-08-11, 10.1038/s41597-026-08061-x | TCIA collection 10.7937/k4qv-hh78 (updated 2026/03/27); nnU-Net weights on Zenodo 10.5281/zenodo.18598644 (**2026-02-10**); GitHub MISAG-BUT/Spinal-Multiple-Myeloma-SEG (v1.0.1 review 2026-02-19) | CC BY 4.0 ("Data Citation Required") | 304.43 GB DICOM (+ NIfTI segmentation package via Aspera) | nnU-Net (Zenodo model zoo record 2026-02-10; Dice values in the paper's full text, not retrievable — nature.com serves only the truncated SharedIt version) | **Earlier data exposure**: Zenodo model + GitHub repo releases in Feb 2026, TCIA DOI record created 2025-06-17 — predates the paper by ~6 months |
| 3 | UCSD-BMETS-Longitudinal | brain, metastases (longitudinal, pre/post treatment) | MRI (T1 post-contrast FSE, DWI/ADC, ASL, SWI; NIfTI) | 300 patients / 1,065 timepoints | lesion masks per timepoint (neuroradiologist-approved) | no journal paper found yet (collection page only) | TCIA 10.7937/5ejb-9g94, **DataCite record created 2026-09-04**, collection updated 2026-10-05 | CC BY 4.0 | 163 GB NIfTI | none published | First public release 2026-09-04 — safely after the 2026-06-30 cutoff; no earlier preprint found |
| 4 | Longitudinal-CT (autoPET/CT-IV) | whole-body, malignant melanoma lesions (primary + mets) | whole-body CT, portal-venous contrast, 3 mm slices, defaced | 300 patients × 2 timepoints (~600 scans) | integer label masks, one label per lesion, per-lesion COG JSON + CSV (location, volume, topology) | challenge paper: Research Square preprint 10.21203/rs.3.rs-10893740/v1, 2026-10-18 (+SSRN 10.2139/ssrn.6841479) | FDAT (Univ. Tübingen) 10.57754/FDAT.qe950-g4h94 v3, **record created 2026-09-18** | CC BY-NC 4.0 (non-commercial) | 51.05 GB zip | challenge baselines (autoPET/CT-IV interactive challenge; nnU-Net-style baselines, Dice per the challenge paper — numbers not verified, preprint is paywalled-free but we did not extract them) | Public release 2026-09-18 (post-cutoff); **caveat**: essentially ALL patients contain at least one `use_for_challenge=True` lesion (verified on data) — the underlying BL/FU scans circulated to registered autoPET/CT-IV challenge participants before the anonymous public release |

### Candidates checked and excluded
- **FDG-PET-CT-Lesions** (TCIA 10.7937/gkr0-xv29, 900 subjects, CT+PT+SEG): updated 2026-08-24 but first released ~2022 (27 citations) — fails the post-2026-06-30 rule.
- **Vestibular-Schwannoma-MC-RC** (TCIA 10.7937/HRZH-2N82): DataCite record 2023-08-09 — fails.
- **SENORA-MRI** (Zenodo 10.5281/zenodo.20773721, stroke/epilepsy lesion masks): first release 2026-06-19 — 11 days before the cutoff; excluded.
- **RCC-AID** (Zenodo 2026-06-16): before cutoff; also classification-oriented.
- **BCSDBT-Seg** (Research Square 2026-08-12, breast tomosynthesis): DBT is quasi-3D (1° acquisition), lesion masks are 2D-ish; weak fit for a 3D interactive harness.

## Per-dataset details

### 1. BEAMSTER — `beamster`
- Citation: Nohel, M., Reguli, S., Kaplanova, R., Jackaninova, J., Chmelik, J., Knybel, L.
  *BEAMSTER: Brain mEtAstases segMentation for STEreotactic Radiotherapy, A Retrospective MRI
  Dataset with Expert Segmentations.* Scientific Data (2026). DOI 10.1038/s41597-026-07777-0.
- Data DOI: 10.6084/m9.figshare.29365844 (figshare, anonymous direct HTTP, 285 files, 4.34 GB).
- Licence: CC BY 4.0 (both paper and data).
- Modality: contrast-enhanced T1 MRI (Siemens Avanto 1.5T / Sola 1.5T / Prisma 3T), rigidly
  registered to the planning-CT space; **defaced**. NIfTI `.nii.gz`.
- Cases/lesions: 140 patients (Dataset_A: 113, Dataset_B: 27 "enriched with small metastases"),
  260 lesions total annotated by an experienced radiation oncologist in the RT planning system.
  Multiple lesions per case; **single binary mask per case** (lesions not instance-labelled).
- Label semantics: binary GT; n_lesions derived as connected components.
- Voxel sizes vary per case: ~0.46–0.62 mm in-plane × 1.0 mm slice (verified on A_001–A_003:
  0.48/0.62/0.53 mm) — the README's 1.0×1.0×1.5 example is not uniform → **1 mm iso
  resampling needed** (per-case, implemented in eval/v4/convert_to_iso.py).
- Published baseline: none verified (full text not accessible; see caveats).
- Caveats: masks annotated for the irradiated metastatic lesion context (Table_clinical_data
  distinguishes primary vs repeat irradiation); very small lesions are the point of Dataset_B.

### 2. Spinal-Multiple-Myeloma-SEG — `spinal_mm_seg`
- Citation: *Spinal-Multiple-Myeloma-SEG: Segmentation of spinal multiple myeloma lesions in
  dual-energy CT.* Scientific Data (2026-08-11). DOI 10.1038/s41597-026-08061-x.
  (MISAG-BUT, Brno University of Technology — same group as BEAMSTER.)
- Data DOI: TCIA collection 10.7937/k4qv-hh78. Licence CC BY 4.0, "Data Citation Required".
- Modality: dual-energy low-dose CT; 576 series: 72 conventional CT (`konv` in SeriesDescription),
  216 VMI (40/80/120 keV), 288 calcium-suppressed (SI 25/50/75/100). 67 patients, 72 scans.
- Lesions: multiple myeloma focal lesions in the spine; masks for lesions (binary) and vertebrae
  (with vertebra-type classification). NIfTI segmentation package exists (Aspera package id 1273).
- Access: DICOM via anonymous TCIA NBIA REST API (`getSeries`/`getImage`, verified working);
  NIfTI package via IBM Aspera faspex public link (scripted access requires a browser-derived
  transfer token — no registration, but not plain HTTP).
- Size: 304.43 GB total (DICOM). Conventional-CT-only subset ≈ 1/8 of series (~38 GB estimated).
- Published baseline: nnU-Net models released on Zenodo (10.5281/zenodo.18598644, 2026-02-10);
  Dice values are in the paper full text, which nature.com currently serves only in truncated
  form — not verified.
- Caveats: **> 300 GB → download stopped per task rule**; earlier data exposure (Zenodo models
  2026-02-10, GitHub 2026-02, TCIA DOI registered 2025-06-17) weakens its "post-cutoff" status,
  though it was explicitly approved by Sameed; MSK tumours are also excluded from the "new
  dataset" rule — this is an *approved* dataset, not an additional one.

### 3. UCSD-BMETS-Longitudinal — `ucsd_bmets`
- Citation: *The University of California San Diego Longitudinal Brain Metastases Annotated
  Multimodal MRI Dataset (UCSD-BMETS-Longitudinal).* TCIA, DOI 10.7937/5ejb-9g94 (2026).
- Modality: MRI; available sequences include T1 post-contrast FSE, DWI/ADC, ASL perfusion,
  susceptibility-weighted; segmentations in NIfTI.
- Cases: 300 patients, 1,065 MRI timepoints (pre- and post-treatment), lesion masks per timepoint.
- Licence: CC BY 4.0. Public, complete.
- Access: **IBM Aspera faspex public package only** (no NBIA DICOM channel). Anonymous, but the
  scripted path needs a transfer token from an Aspera Connect session (env `UCSD_BMETS_BEARER`
  in our downloader) or a manual GUI download. Not downloaded in Phase 4 (blocked).
- Size: 163 GB. Published baseline: none found. First release 2026-09-04 (post-cutoff ✓).

### 4. Longitudinal-CT (autoPET/CT-IV melanoma) — `longitudinal_ct`
- Citation: *Longitudinal-CT*, University Hospital Tübingen, FDAT DOI 10.57754/FDAT.qe950-g4h94
  (v3, 2026-09-18). Challenge paper: *Interactive Lesion Segmentation in Whole-Body PET/CT and
  Longitudinal CT: The autoPET/CT-IV Challenge* (Research Square, 2026-10-18).
- Modality: whole-body CT, portal-venous phase, 3 mm slices, defaced, NIfTI.
- Cases: 300 patients × 2 timepoints (baseline + follow-up); malignant lesions manually
  segmented by two radiologists; integer label masks (per-lesion IDs), lesion COG JSONs,
  per-lesion CSV (location: Lung/Liver/Lymph node/Skeleton/…, volumes, topology, target flag).
- Licence: **CC BY-NC 4.0** (non-commercial — fine for research use; flag if we ever ship models).
- Access: anonymous direct HTTP from FDAT (verified; 51.05 GB single zip + md5). The zip
  unpacks directly into `inputsTr/` (baseline images+masks, COG JSONs, per-patient CSVs)
  and `targetsTr/` (follow-up masks), plus `data_split.json` (the challenge's own train/
  val/test split) and a demographics CSV — layout differs from the FDAT description
  (baseline masks are in inputsTr, not targetsTr; handled in the manifest builder).
- Published baseline: autoPET/CT-IV challenge results (numbers in the preprint; not extracted here).
- Caveats: challenge-exposed subset (see table); 3 mm slices → 1 mm iso resampling inflates data
  ~27× per scan; follow-up scans offer longitudinal evaluation as a phase-5 option.

## Recommended design/test splits (seed 2026, implemented in eval/v4/build_manifest_*.py)

| dataset | design | test | stratification | notes |
|---|---|---|---|---|
| beamster | 30 (24 A + 6 B; half single-/half multi-lesion) | 110 | A/B × single vs multiple lesions | B over-weighted in design for small-lesion stress; per-case split (one mask per case) |
| spinal_mm_seg | 20 patients | 47 patients (52 scans) | none available pre-manifest (lesion counts computed at build) | split by **patient** (some have 2 scans); blocked download |
| ucsd_bmets | 30 patients | 170 patients (fixed random subset) | none (metadata not yet downloadable) | remaining 100 patients marked `unused`; split by patient; one case per timepoint |
| longitudinal_ct | 30 patients | 170 patients (fixed random subset) | none possible — all 300 patients contain autoPET/CT-IV challenge lesions (verified on data) | remaining 100 patients `unused`; baseline scans only; earlier-exposure caveat documented |

For the two very large sets (300 patients each) the test set is a fixed random subset of 170
patients, seed 2026, per the Phase 4 plan (~150–200 cases); every split is patient-disjoint.
