#!/bin/bash

cat <<"EOF"

    _/_/_/    _/_/_/_/  _/        _/_/_/  _/    _/   
   _/    _/  _/        _/          _/    _/  _/      
  _/_/_/    _/_/_/    _/          _/    _/_/         
 _/    _/  _/        _/          _/    _/  _/        
_/    _/  _/_/_/_/  _/_/_/_/  _/_/_/  _/    _/       
                                                     
            ReLiK Inference API

EOF

# pre-download the model if provided in input
if [ "$1" ]; then
    # micromamba run -n base python -c "from relik import Relik; Relik.from_pretrained('$1')"
    python3 -c "from relik import Relik; Relik.from_pretrained('$1')"
fi
