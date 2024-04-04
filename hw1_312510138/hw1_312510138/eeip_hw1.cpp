#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>
#include <utility>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iterator>
#include <cmath>


using namespace std;


string toString(char c) {
    stringstream ss;
    string s;
    ss << c;
    ss >> s;
    return s;
}


// (2 + 4 + 4 + 4) = 14 bytes
#pragma pack(push, 1)
class BmpHeader {

public:
    char signature[2];
	uint32_t filesize;           // total size of bitmap file
	uint16_t reserved1;
	uint16_t reserved2;
	uint32_t pixelDataOffset;    // Start position of pixel data (bytes from the beginning of the file)

    // default constructor
    BmpHeader(){};

    BmpHeader(char s1, char s2, uint32_t fsize, uint16_t r1, uint32_t r2, uint32_t offset) {
        signature[0] = s1;
        signature[1] = s2;
        filesize = fsize;
        reserved1 = r1;
        reserved2 = r2;
        pixelDataOffset = offset;
    }
};


// (4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 4 + 4) = 40 bytes
class BmpInfoHeader {

public:
	uint32_t sizeOfInfoHeader;
    //   (if positive, bottom-up, with origin in lower left corner)
    //   (if negative, top-down, with origin in upper left corner)
	int32_t width;
	int32_t height;

	uint16_t ColorPlanes;   // awlays 1
	uint16_t bitsPerPixel;  // 24 bits

	uint32_t compressionMethod; // 0 or 3 - uncompressed. THIS PROGRAM CONSIDERS ONLY UNCOMPRESSED BMP images
	uint32_t rawBitmapDataSize; // 0 - for uncompressed images

	int32_t xPixelPerMeter{0};
	int32_t yPixelPerMeter{0};

	uint32_t colorTableEntries{0};
	uint32_t importantColors{0};

    // default constructor
    BmpInfoHeader(){};

    BmpInfoHeader(uint32_t s, int32_t w, int32_t h, uint16_t cp, uint16_t bpp, uint32_t cpm, uint32_t rbms, int32_t xpp, int32_t ypp, uint32_t ct, uint32_t ic) {
        sizeOfInfoHeader = s;
        width = w;
        height = h;
        ColorPlanes = cp;
        bitsPerPixel = bpp;
        compressionMethod = cpm;
        rawBitmapDataSize = rbms;
        xPixelPerMeter = xpp;
        yPixelPerMeter = ypp;
        colorTableEntries = ct;
        importantColors = ic;
    }
};
#pragma pack(pop)


class BMP {

public:
    BmpHeader file_header;
    BmpInfoHeader bmp_info_header;
    string mode;    // image manipulation mode
    float ratio;
    string filename;

    // this data with padding information
    vector<uint8_t> data;   // store picture data -- in (B, G, R) order or (B, G, R, A)

    // this data with no padding information
    vector<uint8_t> data_np;

    // constructor -- for reading file
    BMP(const string fname, string m, float f) {
        filename = fname;
        mode = m;
        ratio = f;
        // default do read
        bmp_read(fname);
    }

    // read and write
    void bmp_read(const string fname);
    void bmp_write(const string fname, vector<uint8_t>& outdata, BmpHeader BH, BmpInfoHeader BIH, bool custom_header);

    // show messages
    void showBmpHeader(BmpHeader &header);
    void showBmpInfoHeader(BmpInfoHeader &header);

    // call manipulate functions
    void ImageProcessing() {
        if(mode == "-hf") {
            HorizontalFlip();
        } else if(mode == "-qr") {
            QuantResolution();
        } else if(mode == "-s") {
            Scaling(1.5);
        } else if(mode == "-id") {
            IdentityOut();
        }
        return;
    }

private:
    int width;
    int height;
    int bitsPerPixel;
    int real_image_size; // size with padding

    void write_headers_and_data(ofstream &outfile, vector<uint8_t> &outdata, BmpHeader BH, BmpInfoHeader BIH, bool custom_header);
    void IdentityOut();
    void HorizontalFlip();
    void QuantResolution();
    void Scaling(float ratio);
    void BilinearInterpolate(vector<uint8_t> &indata, vector<uint8_t> &outdata, int OW, int OH, int TW, int TH, int CH);

};


void BMP::showBmpHeader(BmpHeader &header) {
    cout << "===============================================================" << endl;
    cout << "Bmp Header" << endl;
    stringstream ss;
    string s;
    ss << "0x" << std::hex << (int)header.signature[1] << (int)header.signature[0];
    ss >> s;
    cout << "file type: " << s << endl;
    cout << "file size: " << header.filesize << endl;
    cout << "reservedByte: " << std::dec << header.reserved1 << header.reserved2 << endl;
    cout << "Pixel Offset: " << header.pixelDataOffset << endl;
    cout << "===============================================================" << endl;
}


void BMP::showBmpInfoHeader(BmpInfoHeader &header) {
    cout << "===============================================================" << endl;
    cout << "Bmp Info Header" << endl;
    cout << "sizeOfInfoHeader: " << header.sizeOfInfoHeader << endl;
    cout << "width: " << header.width << endl;
    cout << "height: " << header.height << endl;
    cout << "numberOfColorPlanes: " << header.ColorPlanes << endl;
    cout << "bitsPerPixel: " << header.bitsPerPixel << endl;
    cout << "compressionMethod: " << header.compressionMethod << endl;
    cout << "rawBitmapDataSize: " << header.rawBitmapDataSize << endl;
    cout << "xPixelPerMeter: " << header.xPixelPerMeter << endl;
    cout << "yPixelPerMeter: " << header.yPixelPerMeter << endl;
    cout << "colorTableEntries: " << header.colorTableEntries << endl;
    cout << "importantColors: " << header.importantColors << endl;
    cout << "===============================================================" << endl;
}


void BMP::BilinearInterpolate(vector<uint8_t> &indata, vector<uint8_t> &outdata, int OW, int OH, int TW, int TH, int CH) {
    int w_l, h_l, w_h, h_h;
    int a, b, c, d;
    int offset;
    float pixel;
    float dx, dy;
    float w_ratio = float(OW-1) / (TW-1);
    float h_ratio = float(OH-1) / (TH-1);

    for(int y=0; y<TH; y++) {
        for(int x=0; x<TW; x++) {
            for(int k=0; k<CH; k++) {

                w_l = floor(w_ratio * x);
                h_l = floor(h_ratio * y);
                w_h = min(ceil(w_ratio * x), float(OW-1));
                h_h = min(ceil(h_ratio * y), float(OH-1));

                dx = (w_ratio * x) - w_l;
                dy = (h_ratio * y) - h_l;

                a = (int)indata[h_l*OW*CH + w_l*CH + k];   //  a ----- b
                b = (int)indata[h_l*OW*CH + w_h*CH + k];   //  |       |
                c = (int)indata[h_h*OW*CH + w_l*CH + k];   //  |       |
                d = (int)indata[h_h*OW*CH + w_h*CH + k];   //  c ----- d

                pixel = a * (1 - dx) * (1 - dy) + b * (dx) * (1 - dy) + c * (1 - dx) * (dy) + d * (dx) * (dy);
                outdata[y*TW*CH + x*CH + k] = (uint8_t)pixel;
            }
        }
    }
    return;
}


void BMP::write_headers_and_data(ofstream &outfile, vector<uint8_t> &outdata, BmpHeader BH, BmpInfoHeader BIH, bool custom_header) {
    cout << "Show write header:" << endl;
    if(!custom_header) {
        outfile.write((const char*)&file_header, sizeof(file_header));
        outfile.write((const char*)&bmp_info_header, sizeof(bmp_info_header));
        outfile.write((const char*)outdata.data(), outdata.size());
        showBmpHeader(file_header);
        showBmpInfoHeader(bmp_info_header);
    } else {
        outfile.write((const char*)&BH, sizeof(BH));
        outfile.write((const char*)&BIH, sizeof(BIH));
        outfile.write((const char*)outdata.data(), outdata.size());
        showBmpHeader(BH);
        showBmpInfoHeader(BIH);
    }
}


void BMP::bmp_write(const string fname, vector<uint8_t> &outdata, BmpHeader BH, BmpInfoHeader BIH, bool custom_header) {
    ofstream outfile(fname, ios_base::binary);
    if(outfile) {
        write_headers_and_data(outfile, outdata, BH, BIH, custom_header);
    } else {
        cout << "Error! File not correctly written!" << endl;
    }
    cout << "File written successfully!" << endl;
    outfile.close();
}


void BMP::bmp_read(const string fname) {
    ifstream infile(fname, ios_base::binary);

    if(infile) {
        infile.read((char*)&file_header, sizeof(file_header));
        if(file_header.signature[0] != 'B' || file_header.signature[1] != 'M') {
            cerr << "Error: Not a valid BMP file." << endl;
            return;
        }
        infile.read((char*)&bmp_info_header, sizeof(bmp_info_header));

        // store public variable
        width = bmp_info_header.width;
        height = bmp_info_header.height;
        bitsPerPixel = bmp_info_header.bitsPerPixel;

        // compute important data
        int channels = (bitsPerPixel / 8);
        int real_row_width = ceil(width * channels / 4.0) * 4;
        real_image_size = height * real_row_width;

        // jump to the pixel data location
        infile.seekg(file_header.pixelDataOffset, infile.beg);

        // resize the pixel data vector
        data.resize(real_image_size);
        data_np.resize(width * height * channels);

        // store data
        infile.read((char*)data.data(), data.size());

        // give no padding data to data_np
        for(int h=0; h<height; h++) {
            for(int w=0; w<width; w++) {
                for(int c=0; c<channels; c++) {
                    int offset = h*real_row_width + w*channels + c;
                    data_np[h*width*channels + w*channels + c] = data[offset];
                }
            }
        }
    }
    // force pixel offset to 14 + 40 bytes
    file_header.pixelDataOffset = 54;
    // force bmp info header to 40 bytes
    bmp_info_header.sizeOfInfoHeader = 40;
    file_header.filesize = real_image_size + sizeof(BmpHeader) + sizeof(BmpInfoHeader);

    // cout << "Show read header:" << endl;
    // showBmpHeader(file_header);
    // showBmpInfoHeader(bmp_info_header);
    infile.close();
}


// this function might only change the info header size and the filesize
void BMP::IdentityOut() {
    filename.erase(filename.find(".bmp"), 4);
    string s = filename.substr(filename.find("input")+5);
    string ofname = "output" + s + "_id.bmp";
    bmp_write(ofname, data, file_header, bmp_info_header, false);
    return;
}


void BMP::HorizontalFlip() {
    filename.erase(filename.find(".bmp"), 4);
    string s = filename.substr(filename.find("input")+5);
    string ofname = "output" + s + "_flip.bmp";
    vector<uint8_t> data_flip;
    copy(data.begin(), data.end(), back_inserter(data_flip));
    int channels = bitsPerPixel / 8;
    int real_row_width = ceil(width * channels / 4.0) * 4;
    
    // do flip function
    for(int y=0; y<height; ++y) {
        for(int x=0; x<width/2; ++x) {
            int offset_front = y * real_row_width + x * channels;
            int offset_back = y * real_row_width + (width - x - 1) * channels;

            // do swap
            swap(data_flip[offset_front + 0], data_flip[offset_back + 0]);
            swap(data_flip[offset_front + 1], data_flip[offset_back + 1]);
            swap(data_flip[offset_front + 2], data_flip[offset_back + 2]);
            if(channels == 4) {
                swap(data_flip[offset_front + 3], data_flip[offset_back + 3]);
            }
        }
    }

    bmp_write(ofname, data_flip, file_header, bmp_info_header, false);
    return;
}


// quant to 2, 4, 6 bits representations
void BMP::QuantResolution() {
    filename.erase(filename.find(".bmp"), 4);
    string s = filename.substr(filename.find("input")+5);
    string ofname[3];
    ofname[0] = "output" + s + "_1.bmp";   // 6 bits
    ofname[1] = "output" + s + "_2.bmp";   // 4 bits
    ofname[2] = "output" + s + "_3.bmp";   // 2 bits

    vector<uint8_t> data_quant;
    // data_quant.resize(height * width * bitsPerPixel / 8);
    data_quant.resize(real_image_size);
    int channels = bitsPerPixel / 8;
    int real_row_width = ceil(width * channels / 4.0) * 4;
    int quant_rate;

    for(int r=0; r<3; ++r){
        quant_rate = pow(2, r * 2 + 2);
        for(int y=0; y<height; ++y) {
            for(int x=0; x<width; ++x) {
                int offset = y * real_row_width + x * channels; 
                data_quant[offset + 0] = (uint8_t)(((int)data[offset + 0] / quant_rate) * quant_rate);
                data_quant[offset + 1] = (uint8_t)(((int)data[offset + 1] / quant_rate) * quant_rate);
                data_quant[offset + 2] = (uint8_t)(((int)data[offset + 2] / quant_rate) * quant_rate);
                if(channels == 4) {
                    data_quant[offset + 3] = (uint8_t)(((int)data[offset + 3] / quant_rate) * quant_rate);
                }
            }
        }
        bmp_write(ofname[r], data_quant, file_header, bmp_info_header, false);
    }
    return;
}


void BMP::Scaling(float ratio = 1.5) {
    filename.erase(filename.find(".bmp"), 4);
    string s = filename.substr(filename.find("input")+5);

    string ofname_up = "output" + s + "_up.bmp";
    string ofname_down = "output" + s + "_down.bmp";

    // 24 or 32 bits
    int channel = bitsPerPixel / 8;

    // upscale
    int up_width = (int)(width * ratio);
    int up_height = (int)(height * ratio);
    up_width = ceil(up_width / 4.0) * 4;
    // cout << up_width << " " << up_height << endl;
    vector<uint8_t> data_up;
    data_up.resize(up_height * up_width * channel);

    // initialize the upscale header file
    BmpHeader bh_up(
        file_header.signature[0],
        file_header.signature[1],
        data_up.size() + sizeof(BmpHeader) + sizeof(BmpInfoHeader),
        file_header.reserved1,
        file_header.reserved2,
        sizeof(BmpHeader) + sizeof(BmpInfoHeader)
    );

    BmpInfoHeader bih_up(
        sizeof(BmpInfoHeader),
        up_width,
        up_height,
        bmp_info_header.ColorPlanes,
        bmp_info_header.bitsPerPixel,
        bmp_info_header.compressionMethod,
        data_up.size(),
        bmp_info_header.xPixelPerMeter,
        bmp_info_header.yPixelPerMeter,
        bmp_info_header.colorTableEntries,
        bmp_info_header.importantColors
    );

    // start bilinear interpolation
    BilinearInterpolate(data_np, data_up, width, height, up_width, up_height, channel);
    bmp_write(ofname_up, data_up, bh_up, bih_up, true);
    
    // downscale
    int down_width = (int)(width / ratio);
    int down_height = (int)(height / ratio);

    // check padding when down-scaling image
    // The reason is BMP format has a row padding requirement to
    // ensure that each row's length is a multiple of 4 bytes.
    down_width = ceil(down_width / 4.0) * 4;
    // cout << down_width << " " << down_height << endl;

    vector<uint8_t> data_down;
    data_down.resize(down_height * down_width * channel);
    
    // initialize the upscale header file
    BmpHeader bh_down(
        file_header.signature[0],
        file_header.signature[1],
        data_down.size() + sizeof(BmpHeader) + sizeof(BmpInfoHeader),
        file_header.reserved1,
        file_header.reserved2,
        sizeof(BmpHeader) + sizeof(BmpInfoHeader)
    );

    BmpInfoHeader bih_down(
        sizeof(BmpInfoHeader),
        down_width,
        down_height,
        bmp_info_header.ColorPlanes,
        bmp_info_header.bitsPerPixel,
        bmp_info_header.compressionMethod,
        data_down.size(),
        bmp_info_header.xPixelPerMeter,
        bmp_info_header.yPixelPerMeter,
        bmp_info_header.colorTableEntries,
        bmp_info_header.importantColors
    );

    // start bilinear interpolation
    BilinearInterpolate(data_np, data_down, width, height, down_width, down_height, channel);
    bmp_write(ofname_down, data_down, bh_down, bih_down, true);

    return;
}


int main(int argc, char** argv) {

    // input image file
    string infile = argv[1];

    // get manipulation mode
    string mode = argv[2];

    // for scaling 
    float ratio = 0;
    if(argc == 3) {
        stringstream ss;
        float f;
        ss << argv[3];
        ss >> f;
        ratio = f;
    }

    // constructor
    BMP bmp(infile, mode, ratio);

    // do image processing
    bmp.ImageProcessing();

    return 0;
}


