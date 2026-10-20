# SafeDrive-R

SafeDrive-R is a local Responsible AI research prototype for pedestrian detection in BDD100K road scenes. It combines a pretrained detector with an environmental audit and controlled robustness checks.

## What is included

- Pretrained YOLO person detection for uploaded or BDD100K validation images.
- BDD100K validation-set profiles for time of day, weather, and scene.
- A deterministic day/night evaluation sample with precision, recall, false positives, false negatives, and false-negative rate.
- Per-image brightness and center-occlusion stress tests.
- CSV export for measured day/night evaluation metrics.

The metrics depend on the model, confidence threshold, sample size, and IoU threshold. They are sample estimates, not official benchmark mAP. BDD100K environmental groups are used here; this dataset does not justify unverified demographic fairness claims.

## Dataset layout in this workspace

The app defaults to the existing files and does not copy the dataset:

```text
bdd100k/bdd100k/images/100k/val/
bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json
```

The validation JSON is about 208 MB and includes image-level attributes and object bounding boxes. The workspace currently lacks the BDD100K object-detection training annotations, so model fine-tuning is not part of this implementation. The segmentation labels are a separate task and are not used as pedestrian boxes.

## Run on Windows

From the workspace root, create and activate a virtual environment, then install the dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The first detector run downloads the selected pretrained YOLO weights and therefore needs internet access. A GPU is optional; CPU inference works but will be slower. The app is for coursework and is not suitable for vehicle control or safety-critical use.

## Experiment discipline

Keep the model name, confidence threshold, sample size, and IoU threshold with any reported numbers. Do not describe brightness or occlusion checks as successful adversarial attacks. The dashboard currently implements controlled corruption/occlusion stress tests; white-box attacks, adversarial training, formal XAI, and model fine-tuning remain future experiments that need separate validation.

Run the metric unit tests with:

```powershell
python -m unittest discover -s tests -v
```

## Publish on GitHub and deploy to Streamlit Community Cloud

The repository intentionally excludes the BDD100K folders and `.pt` model weights. GitHub is not a suitable place for the dataset, and Ultralytics can download the selected YOLO weights the first time inference runs. Without the local BDD100K files, the hosted app still supports uploaded-image detection; dataset profiles, validation-image selection, and day/night evaluation require the annotations and images to be made available separately.

1. Install [Git for Windows](https://git-scm.com/download/win), then create an empty repository on [GitHub](https://github.com/new). Do not initialize it with a README, license, or `.gitignore`.
2. In PowerShell, from this project folder, run the following commands. Replace the remote URL with the HTTPS URL of your new GitHub repository.

	```powershell
	git init
	git add .gitignore README.md app.py requirements.txt src tests
	git commit -m "Prepare SafeDrive-R Streamlit app"
	git branch -M main
	git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
	git push -u origin main
	```

3. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and authorize access to the repository.
4. Choose **Create app**, select the repository and `main` branch, and set the app file path to `app.py`. Deploy the app.
5. Open the generated app URL. Uploaded-image detection is available without BDD100K. To enable the dataset audits in the cloud, host the data separately and update the annotation and image paths in the app configuration; do not add the dataset to this Git repository.

If GitHub prompts for authentication in the terminal, complete its browser-based sign-in flow. Never commit access tokens, passwords, or `.streamlit/secrets.toml`.
