import os
from subprocess import Popen, PIPE
import interfile

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

def empty_kex_recon_command():
    """ has no paths specified"""
    cmd_str = "e7_recon "
    cmd_str += " --tof "
    cmd_str += " --mash4 "
    cmd_str += " --algo op-osem "
    cmd_str += " --is 3,21 "
    cmd_str += " -n " 
    # cmd_str += " --offs -1.156,0.198,-754.152,-0.091,0.03,-0.777 " #hdr copy
    cmd_str += " --gf " 
    cmd_str += " --quant 1 "
    cmd_str += " -w 200 "
    cmd_str += " --fltr GAUSSIAN,5,5 "
    cmd_str += " -l 72 " 
    cmd_str += " --fl "
    cmd_str += " --ecf "
    cmd_str += " --izoom 1 "
    cmd_str += " --force "
    # cmd_str += " --cvrg 97 "  #hdr copy
    cmd_str += " --rs "
    cmd_str += " -e " 
    cmd_str += " --oi " 
    # cmd_str += " --dcr {} ".format(dcr) #varies between headers
    # cmd_str += " --reglt -249.512,-371.012" #hdr copy
    # cmd_str += " --bp 42.986000 "
    return cmd_str

def kex_recon(efile, nfile, oifile, verbose=True, gapfill=True, dcr=None, izoom=1, model=None):
    """ kex_recon: run basic optof e7tools reconstructions with parameters from kex data header.
    input: efile,  nfile, oifile
    note that the oifile file format is determined by e7recon, according to efile and nfile. 
    """
    cmd_str = "e7_recon "
    cmd_str += " --tof "
    cmd_str += " --mash4 "
    cmd_str += " --algo op-osem "
    cmd_str += " --is 3,21 "
    cmd_str += " -n " + nfile
    cmd_str += " --offs -1.156,0.198,-754.152,-0.091,0.03,-0.777 " #keep
    if gapfill:
        cmd_str += " --gf " 
    cmd_str += " --quant 1 "
    cmd_str += " -w 200 "
    cmd_str += " --fltr GAUSSIAN,5,5 "
    cmd_str += " -l 72 " if verbose else " -l 0 "
    cmd_str += " --fl "
    cmd_str += " --ecf "
    if model is not None:
        cmd_str += " --model {} ".format(model)

    cmd_str += " --izoom {} ".format(izoom)
    cmd_str += " --force "
    cmd_str += " --cvrg 97 " #keep
    cmd_str += " --rs "
    cmd_str += " -e " + efile
    cmd_str += " --oi " + oifile
    if dcr is not None:
        cmd_str +=  " --dcr {} ".format(dcr)
    cmd_str += " --reglt -249.512,-348.512 " # free
    cmd_str += " --bp 801.486000 " # free
    return run_tool(command=cmd_str)

def recon_intermediates(emission_path, 
                        norm_path, 
                        output_emission_path, 
                        output_debug_path=None, tof=True, gapfill=False):
    cmd_str = "e7_recon.exe "
    if tof:
        cmd_str += " --tof "
    if gapfill:
        cmd_str += " --gf "
    cmd_str += " -e " + emission_path
    cmd_str += " --oe " + output_emission_path
    cmd_str += " -n " + norm_path
    cmd_str += " --force"
    # cmd_str += " --dcr 1"
    if output_debug_path is not None:
        cmd_str += " -d " + output_debug_path
    return run_tool(command = cmd_str)



