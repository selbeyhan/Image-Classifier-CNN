# Image-Classifier-CNN

1. Make a virtaul environment, call it venv, 
python3 -m venv venv

2. Install the following modules

Mac M1/M2/M3
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Mac GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

2. Place the train.pkl and val.pkl file in the directory
3. Make a model.pth file in the directory
4. Run main.py, this uses the best model we have to use
python3 main.py
