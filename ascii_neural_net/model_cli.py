from __future__ import annotations
import argparse
from .models import load_model
def main():
 p=argparse.ArgumentParser(); p.add_argument("model"); p.add_argument("values",nargs="*",type=float); a=p.parse_args(); network,_=load_model(a.model); print(network.predict(a.values))
if __name__=="__main__": main()
