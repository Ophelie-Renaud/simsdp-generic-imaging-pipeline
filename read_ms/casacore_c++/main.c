// main.c
#include <stdio.h>
#include "read_ms.h"

extern void read_ms(const char* filename);

int main() {
    //read_ms("../sim_small.ms");
    read_ms("/home/orenaud/Desktop/nancep/VirA-SB155/SB155.rebin.MS");
    return 0;
}

