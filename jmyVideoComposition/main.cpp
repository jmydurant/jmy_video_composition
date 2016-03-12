#include "main.h"

int main(int argc, char ** argv) {
	fprintf(stdout, "let's start!!\n");

	// step 1: read all the video file and ini file.
	// argv should have the path of the file.
	if (argc == 1) {
		fprintf(stderr, "no file found!!\n\n niconiconi~~~\n");
		return 1;
	}
	else {
		fprintf(stdout, "file path is %s\n", argv[1]);
	}
	string myPath(argv[1]);
	if (myconfig.readAllTheParam(myPath) != 0) {
		return 1;
	}

	// prepare all the video the program need...
	/*initModify initmodify;
	initmodify.exec(myconfig);*/

	// use MVC to clone the image...and make the image smooth...

	MVCCore mvcCore;
	mvcCore.exec(myconfig);
	
	waitKey(0);
	return 0;
}