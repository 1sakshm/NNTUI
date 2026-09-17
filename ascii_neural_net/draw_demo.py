from __future__ import annotations
import argparse
from .canvas import run_canvas
from .mnist import load_mnist, train_mnist
def main():
 p=argparse.ArgumentParser(); p.add_argument("--data",default="data/mnist"); p.add_argument("--limit",type=int,default=500); p.add_argument("--epochs",type=int,default=3); a=p.parse_args(); x,y=load_mnist(a.data,limit=a.limit); run_canvas(train_mnist(x,y,epochs=a.epochs))
if __name__=="__main__": main()
