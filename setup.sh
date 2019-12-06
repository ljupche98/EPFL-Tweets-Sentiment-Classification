echo "Creating environment"
python3 -m venv env
echo "Activation environment"
source env/bin/activate
echo "Installing packages"
pip3 install -r requirements.txt
echo "Cloning sent2vec"
git clone https://github.com/epfml/sent2vec.git
(
    cd sent2vec
    echo "Installing sent2vec"
    pip3 install .
    make
)