import os
from subprocess import Popen, PIPE

def run_tool(command, e7folder=None, verbose = True):
    """ run_tool: runs the command string in the e7tools folder. Prints program standard output if verbose.
    input: command, e7folder=None, verbose = True
    return e7tools returncode """
        
    if e7folder is None:
        # e7 folder on author's laptop
        e7folder = r"C:/Users/petct/Desktop/Siemense7tools/Siemense7tools/C-Siemens-PET-VG60/bin.win64-VG60/"
   
    process = Popen(e7folder  + command, stdout=PIPE, stderr=PIPE)
    
    if verbose:
        stdout, stderr = process.communicate()
        print("stdout\n", str(stdout).replace(r"\r\n", "\n"))
        print("stderr\n", str(stderr).replace(r"\r\n", "\n"))
        
    return process.returncode

def recon(efile, nfile, oifile):
    """ recon: run basic optof e7tools reconstructions with parameters from kex data header.
    input: efile,  nfile, oifile
    note that the oifile file format is determined by e7recon, according to efile and nfile. 
    """

    cmd_str = "e7_recon "
    
    cmd_str += " --tof "
    cmd_str += " --mash4 "
    cmd_str += " --algo op-osem "
    cmd_str += " --is 3,21 "
    cmd_str += " -n " + nfile
    cmd_str += " --gf "
    cmd_str += " --quant 1 "
    cmd_str += " -w 200 "
    
    cmd_str += " --fltr GAUSSIAN,5,5 "
    cmd_str += " -l 0 "
    cmd_str += " --fl "
    cmd_str += " --ecf "
    cmd_str += " --izoom 1 "
    cmd_str += " --force "
    cmd_str += " --rs "
    
    cmd_str += " -e " + efile 
    cmd_str += " --oi " + oifile

    run_tool(command=cmd_str)