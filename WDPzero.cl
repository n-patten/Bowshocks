procedure WDPzero(infile)

# task to overscan subtract and trim WIRO DoublePrime images from WIRO
# version 1.0 by Chip K. 2015Oct07

  string infile
  string outnam
  string cline
  
begin
   string cstr
   struct line
   
   cline="INDEF"
    imred
    bias
   print ("overscan zero subtraction script for WIRO Double Prime imager")   
   print ("Input file names must end in .fit and _z will be appended")   
   list = infile
   while (fscan(list, line) != EOF) {
        imcopy (line//"[8:2100,1:2048]", "foo1.fits", ver-)
        colbias ("foo1.fits", "foo1b.fits", bias="[2054:2089,1:2048]", trim="[1:2048,*]", interac-, order=3)
        imcopy (line//"[2101:4193,1:2048]", "foo2.fits", ver- )
        colbias ("foo2.fits", "foo2b.fits", bias="[4:40,1:2048]", trim="[46:2093,*]", interac-, order=3)
        imcopy (line//"[8:2100,2049:4096]", "foo3.fits", ver-)
        colbias ("foo3.fits", "foo3b.fits", bias="[2054:2088,1:2048]", trim="[1:2048,*]", interac-, order=3)
        imcopy (line//"[2101:4193,2049:4096]", "foo4.fits", ver- )
        colbias ("foo4.fits", "foo4b.fits", bias="[4:40,1:2048]", trim="[46:2093,*]", interac-, order=3)
	outnam=substr(line,1,strlen(line)-4)//"_z.fits"
	print (outnam)
	
        imcopy (line//"[1:4096,1:4096]", "foo5.fits",ver-)
	imarith ("foo5.fits", "*", 1.0, "foo6.fits")
        imcopy ("foo1b.fits", "foo6.fits[1:2048,1:2048]", ver-)
        imcopy ("foo2b.fits", "foo6.fits[2049:4096,1:2048]", ver-)
        imcopy ("foo3b.fits", "foo6.fits[1:2048,2049:4096]", ver-)
        imcopy ("foo4b.fits", "foo6.fits[2049:4096,2049:4096]", ver-)
	imcopy ("foo6.fits", outnam,ver- )
        delete ("foo*.fits", ver-)
   }


end



