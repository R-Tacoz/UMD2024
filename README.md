# hey

When running an evaluation file, cd into the folder that it's in since most paths are relative.

Some notes on Python environment:
* Install requirements.txt using pip
* Make note that peft and transformers have edits, thus are not included in requirements. Import these using "import sys; sys.path.append(path_to_pkgs)"
* ot_pytorch comes from https://github.com/rythei/PyTorchOT/blob/master/ot_pytorch.py