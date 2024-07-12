# Create a new conda environment 
conda remove -n fanasr2024 --all   
conda create -n funasr2024 python=3.9   
conda activate funasr2024   

# Install the required packages  
pip3 install -e ./   
pip3 install torch torchvision torchaudio   
pip install hdbscan   
pip install -U rotary_embedding_torch   
pip install ffmpeg-python   
sudo apt-get update   
sudo apt-get install sox   
pip install sox   
pip install tensorboardX   
pip install -U transformers  

# Install SOX  
cd /usr/lib/x86_64-linux-gnu/   
sudo cp libsox.so.3 libsox.so   


