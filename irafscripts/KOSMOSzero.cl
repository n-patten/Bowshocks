procedure KOSMOSzero(infile)

# task to overscan subtract and trim KOSMOS images from APO
# version 1.0 by Chip K. 2015Oct07 and Nikhil

  string infile
  string outnam
  string cline
  
begin
   string cstr
   struct line
   
   cline="INDEF"
    imred
    bias
   print ("overscan zero subtraction script for KOSMOS spectrograph")   
   print ("Input file names must end in .fit and _z will be appended")   
   list = infile
   while (fscan(list, line) != EOF) {
        imcopy (line//"[1:1124,*]", "foo1.fits", ver-)
        colbias ("foo1.fits", "foo1b.fits", bias="[1025:1074,*]", trim="[150:512,*]", interac-, order=3)
        imcopy (line//"[1:1124,*]", "foo2.fits", ver- )
        colbias ("foo2.fits", "foo2b.fits", bias="[1075:1124,*]", trim="[513:846,*]", interac-, order=3)
	outnam=substr(line,1,strlen(line)-4)//"_z.fits"
	print (outnam)
	
        imcopy (line//"[1:1024,1:4096]", "foo5.fits",ver-)
	imarith ("foo5.fits", "*", 1.0, "foo6.fits")
        imcopy ("foo1b.fits", "foo6.fits[150:512,*]", ver-)
        imcopy ("foo2b.fits", "foo6.fits[513:846,*]", ver-)
	imcopy ("foo6.fits", outnam,ver- )
        delete ("foo*.fits", ver-)
   }


end



