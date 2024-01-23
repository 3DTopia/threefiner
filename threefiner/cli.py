import os
import tyro
from threefiner.opt import config_defaults, config_doc, check_options
from threefiner.gui import GUI


def main():    
    opt = tyro.cli(tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc))
    opt = check_options(opt)
    gui = GUI(opt)
    if gui.gui:
        gui.render()
    else:
        gui.train(opt.iters)

if __name__ == "__main__":
    main()
