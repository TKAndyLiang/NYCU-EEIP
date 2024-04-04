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
    float beta;
    float kernel_size;
    string filename;
    string ofilename;

    // this data with padding information
    vector<uint8_t> data;   // store picture data -- in (B, G, R) order or (B, G, R, A)

    // this data with no padding information
    vector<uint8_t> data_np;

    // constructor -- for reading file
    BMP(const string fname, const string ofname, string m, float f, float b, float k) {
        filename = fname;
        ofilename = ofname;
        mode = m;
        ratio = f;
        beta = b;
        kernel_size = k;
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
            Scaling(ratio);
        } else if(mode == "-id") {
            IdentityOut();
        } else if(mode == "-plt") {
            float r = (ratio == 0) ? 0.5 : ratio;
            PowerLawTransformation(r);
        } else if(mode == "-heq") {
            HistogramEQ();
        } else if(mode == "-bca") {
            float r = (ratio == 0) ? 3.0 : ratio;
            BrightContrastAdjustment(r, beta);
        } else if(mode == "-lgf") {
            LaplacianGaussianFilter();
        } else if(mode == "-gs") {
            int k = (beta == 0) ? 5 : beta; // here beta stand for kernel_size
            double s = (ratio == 0) ? 1.0 : ratio;
            GaussianSmooth(k, s);
        } else if(mode == "-bf") {
            int k = (kernel_size == 0) ? 11 : kernel_size;
            double r = (ratio == 0) ? 2.0 : ratio;
            double b = (beta == 0) ? 150.0 : beta;
            BilateralFilter(k, r, b);
        } else if(mode == "-gw") {
            double sb_bias = ratio;
            double sg_bias = beta;
            double sr_bias = kernel_size; // here kernel size stand for the bias
            // cout << sb_bias << " " << sg_bias << " " << sr_bias << endl;
            GrayWorld(sb_bias, sg_bias, sr_bias);
        } else if(mode == "-maxrgb") {
            double sb_bias = ratio;
            double sg_bias = beta;
            double sr_bias = kernel_size; // here kernel size stand for the bias
            // cout << sb_bias << " " << sg_bias << " " << sr_bias << endl;
            MaxRGB(sb_bias, sg_bias, sr_bias);
        }

        // default
        else {
            IdentityOut();
        }
        return;
    }

private:
    int width;
    int height;
    int bitsPerPixel;
    int real_image_size; // size with padding

    // eeip_hw1 part
    void write_headers_and_data(ofstream &outfile, vector<uint8_t> &outdata, BmpHeader BH, BmpInfoHeader BIH, bool custom_header);
    void IdentityOut();
    void HorizontalFlip();
    void QuantResolution();
    void Scaling(float ratio);
    void BilinearInterpolate(vector<uint8_t> &indata, vector<uint8_t> &outdata, int OW, int OH, int TW, int TH, int CH);

    // eeip_hw2 part
    // low luminosity enhancement method1
    void PowerLawTransformation(float ratio);

    // low luminosity enhancement method2
    void HistogramEQ();
    void BrightContrastAdjustment(float alpha, float beta);

    // sharpness enhancement method1 and method2 with different kernels
    void LaplacianGaussianFilter();

    // image denoising method1
    void GaussianSmooth(int kernel_size, double sigma);

    // image denoising method2 -> s for spatial, c for color
    void BilateralFilter(int kernel_size, double sigma_s, double sigma_c);

    // color constancy -> Gray World
    void GrayWorld(double sb_bias, double sg_bias, double sr_bias);

    // color constancy -> maxRGB
    void MaxRGB(double sb_bias, double sg_bias, double sr_bias);
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
    // filename.erase(filename.find(".bmp"), 4);
    // string s = filename.substr(filename.find("input")+5);
    // string ofname = "output" + s + "_id.bmp";
    string ofname = ofilename + "_id.bmp";
    bmp_write(ofname, data, file_header, bmp_info_header, false);
    return;
}


void BMP::HorizontalFlip() {
    // filename.erase(filename.find(".bmp"), 4);
    // string s = filename.substr(filename.find("input")+5);
    // string ofname = "output" + s + "_flip.bmp";
    string ofname = ofilename + "_flip.bmp";
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
    // filename.erase(filename.find(".bmp"), 4);
    // string s = filename.substr(filename.find("input")+5);
    // string ofname[3];
    // ofname[0] = "output" + s + "_1.bmp";   // 6 bits
    // ofname[1] = "output" + s + "_2.bmp";   // 4 bits
    // ofname[2] = "output" + s + "_3.bmp";   // 2 bits
    string ofname[3];
    ofname[0] = ofilename + "_1.bmp";   // 6 bits
    ofname[1] = ofilename + "_2.bmp";   // 4 bits
    ofname[2] = ofilename + "_3.bmp";   // 2 bits

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
    // filename.erase(filename.find(".bmp"), 4);
    // string s = filename.substr(filename.find("input")+5);
    // string ofname_up = "output" + s + "_up.bmp";
    // string ofname_down = "output" + s + "_down.bmp";
    string ofname_up = ofilename + "_up.bmp";
    string ofname_down = ofilename + "_down.bmp";

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


void BMP::PowerLawTransformation(float ratio = 0.5) {
    // filename.erase(filename.find(".bmp"), 4);
    // string s = filename.substr(filename.find("input")+5);
    // string ofname = "output" + s + "_1.bmp";
    string ofname = ofilename;

    vector<uint8_t> data_plt;
    data_plt.resize(real_image_size);
    int channels = bitsPerPixel / 8;
    int real_row_width = ceil(width * channels / 4.0) * 4;
    float pixel, pixel_n;

    // do transformation function
    for(int y=0; y<height; ++y) {
        for(int x=0; x<width; ++x) {
            for(int k=0; k<channels; k++) {
                int offset = y * real_row_width + x * channels + k;
                pixel_n = (float)data[offset] / 255.0;
                pixel = 255 * pow(pixel_n, ratio);
                data_plt[offset] = (uint8_t)pixel;
            }
        }
    }

    bmp_write(ofname, data_plt, file_header, bmp_info_header, false);

    return;
}


void BMP::HistogramEQ() {
    // filename.erase(filename.find(".bmp"), 4);
    // string s = filename.substr(filename.find("input")+5);
    // string ofname = "output" + s + "_eq.bmp";
    string ofname = ofilename + "_eq.bmp";

    vector<uint8_t> data_heq;
    copy(data.begin(), data.end(), back_inserter(data_heq));
    int channels = bitsPerPixel / 8;
    int real_row_width = ceil(width * channels / 4.0) * 4;

    // build histogram
    vector<int> hist(256, 0);
    vector<int> eq_lut(256, 0);
    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            for(int k=0; k<channels; k++) {
                int offset = y * real_row_width + x * channels + k;
                int value = (int)data_heq[offset];
                hist[value] += 1;
            }
        }
    }

    // find first non-zero bin
    int id = 0, sum = 0;
    while(hist[id] == 0)    id++;
    // compute scale
    float scale = 255.0 / ((height * width * channels) - (hist[id]));
    id++;
    // prepare equalization look up table
    for(int i=id; i<hist.size(); i++) {
        sum += hist[i];
        eq_lut[i] = max(0, min((int)round(sum * scale), 255));
    }

    // do eq
    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            for(int k=0; k<channels; k++) {
                int offset = y * real_row_width + x * channels + k;
                int value = (int)data_heq[offset];
                data_heq[offset] = (uint8_t)eq_lut[value];
            }
        }
    }

    bmp_write(ofname, data_heq, file_header, bmp_info_header, false);

    return;
}


void BMP::BrightContrastAdjustment(float alpha = 3.0, float beta = 10) {
    // filename.erase(filename.find(".bmp"), 4);
    // string s = filename.substr(filename.find("input")+5);
    // string ofname = "output" + s + "_2.bmp";
    string ofname = ofilename;

    vector<uint8_t> data_bca;
    data_bca.resize(real_image_size);
    int channels = bitsPerPixel / 8;
    int real_row_width = ceil(width * channels / 4.0) * 4;

    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            for(int k=0; k<channels; k++) {
                int offset = y * real_row_width + x * channels + k;
                int pixel = max(0, min((int)(alpha * (int)data[offset] + beta), 255));
                data_bca[offset] = (uint8_t)pixel;
            }
        }
    }

    bmp_write(ofname, data_bca, file_header, bmp_info_header, false);

    return;
}


inline double LOG(double sigma, int x, int y) {
    double r = x * x + y * y;
    double dev = 2.0 * sigma * sigma;
    return ((1.0) / (M_PI * pow(sigma, 4.0))) * (1.0 - (r / dev)) * (exp((-1 * r) / (dev)));
}


void BMP::LaplacianGaussianFilter() {
    // filename.erase(filename.find(".bmp"), 4);
    // string s = filename.substr(filename.find("input")+5);
    // string ofname[2];
    // ofname[0] = "output" + s + "_1.bmp";
    // ofname[1] = "output" + s + "_2.bmp";
    string ofname[2];
    ofname[0] = ofilename + "_1_log.bmp";
    ofname[1] = ofilename + "_2_log.bmp";

    int N = 5;
    int NN = pow(N, 2);
    int padding = N / 2;

    int lg_kernel[2][NN] = {
        {0,  0,  -1,  0,  0,
         0, -1,  -2, -1,  0,
         -1, -2, 17, -2, -1,
         0, -1,  -2, -1,  0,
         0,  0,  -1,  0,  0},

        { 0, -1, -2, -1,  0,
         -1, -2,  2, -2, -1,
         -2,  2, 17,  2, -2,
         -1, -2,  2, -2, -1,
          0, -1, -2, -1,  0}
    };

    // int lg_kernel[NN] = { 0, -1, 0, -1, 5, -1, 0, -1, 0 };

    vector<uint8_t> data_lgf;
    data_lgf.resize(real_image_size);
    int channels = bitsPerPixel / 8;
    int real_row_width = ceil(width * channels / 4.0) * 4;

    for(int m=0; m<2; m++) {
        for(int y=0; y<height; y++) {
            for(int x=0; x<width; x++) {
                // edge do not compute
                if(y < padding || y >= (height - padding) || x < padding || x >= (width - padding)) {
                    int store_offset = y * real_row_width + x * channels;
                    data_lgf[store_offset + 0] = (uint8_t)data[store_offset + 0];
                    data_lgf[store_offset + 1] = (uint8_t)data[store_offset + 1];
                    data_lgf[store_offset + 2] = (uint8_t)data[store_offset + 2];
                    if(channels == 4) {
                        data_lgf[store_offset + 3] = (uint8_t)data[store_offset + 3];
                    }
                }
                else {
                    int store_offset = y * real_row_width + x * channels;
                    int temp[4] = {0, 0, 0, 0};
                    int cnt = 0;
                    for(int ky=y-padding; ky<(y+N-padding); ky++) {
                        for(int kx=x-padding; kx<(x+N-padding); kx++) {
                            int offset = ky * real_row_width + kx * channels;
                            temp[0] += (int)data[offset + 0] * lg_kernel[m][cnt];
                            temp[1] += (int)data[offset + 1] * lg_kernel[m][cnt];
                            temp[2] += (int)data[offset + 2] * lg_kernel[m][cnt];
                            if(channels == 4) {
                                temp[3] += (int)data[offset + 3] * lg_kernel[m][cnt];
                            }
                            cnt++;
                        }
                    }
                    for(int i=0; i<4; i++) {
                        temp[i] = max(0, min(temp[i], 255));
                    }
                    data_lgf[store_offset + 0] = (uint8_t)temp[0];
                    data_lgf[store_offset + 1] = (uint8_t)temp[1];
                    data_lgf[store_offset + 2] = (uint8_t)temp[2];
                    if(channels == 4) {
                        data_lgf[store_offset + 3] = (uint8_t)temp[3];
                    }
                }
            }
        }
        bmp_write(ofname[m], data_lgf, file_header, bmp_info_header, false);
    }

    return;
}


void BMP::GaussianSmooth(int kernel_size = 5, double sigma = 1.0) {
    // filename.erase(filename.find(".bmp"), 4);
    // string s = filename.substr(filename.find("input")+5);
    // string ofname = "output" + s + "_1.bmp";
    string ofname = ofilename;

    int N = kernel_size;
    int NN = pow(N, 2);
    double lg_kernel[NN] = {0};
    int padding = N / 2;

    vector<uint8_t> data_gs;
    data_gs.resize(real_image_size);
    int channels = bitsPerPixel / 8;
    int real_row_width = ceil(width * channels / 4.0) * 4;

    double r, sum = 0;
    double dev = 2.0 * sigma * sigma;
    
    // generate Gaussian kernel
    for(int x=-padding; x<padding+1; x++) {
        for(int y=-padding; y<padding+1; y++) {
            int offset = (x + padding) * N + (y + padding);
            r = (x * x + y * y);
            lg_kernel[offset] = (exp((-1) * r / dev)) / (M_PI * dev);
            sum += lg_kernel[offset];
        }
    }

    // normalize kernel
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            lg_kernel[i*N+j] /= sum;
            // cout << lg_kernel[i*N+j] << "\t";
        }
        // cout << endl;
    }

    // do Gaussian smoothing
    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            int store_offset = y * real_row_width + x * channels;
            double temp[4] = {0, 0, 0, 0};
            int cnt = 0;
            for(int ky=y-padding; ky<(y+N-padding); ky++) {
                for(int kx=x-padding; kx<(x+N-padding); kx++) {
                    if(ky < 0 || ky >= height || kx < 0 || kx >= width) {
                        cnt++;
                        continue;
                    }
                    else {
                        int offset = ky * real_row_width + kx * channels;
                        temp[0] += (int)data[offset + 0] * lg_kernel[cnt];
                        temp[1] += (int)data[offset + 1] * lg_kernel[cnt];
                        temp[2] += (int)data[offset + 2] * lg_kernel[cnt];
                        if(channels == 4) {
                            temp[3] += (int)data[offset + 3] * lg_kernel[cnt];
                        }
                        cnt++;
                    }
                }
            }
            for(int i=0; i<4; i++) {
                temp[i] = max(0.0, min(temp[i], 255.0));
            }
            data_gs[store_offset + 0] = (uint8_t)temp[0];
            data_gs[store_offset + 1] = (uint8_t)temp[1];
            data_gs[store_offset + 2] = (uint8_t)temp[2];
            if(channels == 4) {
                data_gs[store_offset + 3] = (uint8_t)temp[3];
            }
        }
    }

    bmp_write(ofname, data_gs, file_header, bmp_info_header, false);
    return;
}


void BMP::BilateralFilter(int kernel_size = 11, double sigma_s = 2.0, double sigma_c = 150.0) {
    // filename.erase(filename.find(".bmp"), 4);
    // string s = filename.substr(filename.find("input")+5);
    // string ofname = "output" + s + "_2.bmp";
    string ofname = ofilename;

    int N = kernel_size;
    int NN = pow(N, 2);
    double space_kernel[NN] = {0};      // Gaussian kernel
    double color_factor[256] = {0};     // ColorFactor table
    int padding = N / 2;

    vector<uint8_t> data_bf;
    data_bf.resize(real_image_size);
    int channels = bitsPerPixel / 8;
    int real_row_width = ceil(width * channels / 4.0) * 4;

    // prepare Gaussian kernel
    for(int x=-padding; x<padding+1; x++) {
        for(int y=-padding; y<padding+1; y++) {
            int offset = (x + padding) * N + (y + padding);
            space_kernel[offset] = exp(((-1) * (x * x + y * y)) / (2 * sigma_s * sigma_s));
            // cout << std::left << space_kernel[offset] << " " << setw(13);
        }
        // cout << endl;
    }

    // prepare color factor
    for(int i=0; i<256; i++) {
        double diff = i / sigma_c;
        color_factor[i] = exp(-0.5 * pow(diff, 2));
    }

    // do Bilateral Filtering
    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            int store_offset = y * real_row_width + x * channels;   // centor of the kernel
            double temp[4] = {0, 0, 0, 0};
            double weight[4] = {0, 0, 0, 0};
            double sum[4] = {0, 0, 0, 0};
            int diff[4] = {0, 0, 0, 0};
            int cnt = 0;

            // do bilateral filtering
            for(int ky=y-padding; ky<(y+N-padding); ky++) {
                for(int kx=x-padding; kx<(x+N-padding); kx++) {
                    int cal_offset = ky * real_row_width + kx * channels;
                    if(ky < 0 || ky >= height || kx < 0 || kx >= width) {
                        cnt++;
                        continue;
                    }
                    else {
                        diff[0] = abs((int)data[store_offset + 0] - (int)data[cal_offset + 0]);
                        diff[1] = abs((int)data[store_offset + 1] - (int)data[cal_offset + 1]);
                        diff[2] = abs((int)data[store_offset + 2] - (int)data[cal_offset + 2]);

                        weight[0] = space_kernel[cnt] * color_factor[diff[0]];
                        weight[1] = space_kernel[cnt] * color_factor[diff[1]];
                        weight[2] = space_kernel[cnt] * color_factor[diff[2]];

                        temp[0] += (int)data[cal_offset + 0] * weight[0];
                        temp[1] += (int)data[cal_offset + 1] * weight[1];
                        temp[2] += (int)data[cal_offset + 2] * weight[2];

                        sum[0] += weight[0];
                        sum[1] += weight[1];
                        sum[2] += weight[2];

                        if(channels == 4) {
                            diff[3] = abs((int)data[store_offset + 3] - (int)data[cal_offset + 3]);
                            weight[3] = space_kernel[cnt] * color_factor[diff[3]];
                            temp[3] += (int)data[cal_offset + 3] * weight[3];
                            sum[3] += weight[3];
                        }
                        cnt++;
                    }
                }
            }

            // edge judge
            for(int i=0; i<4; i++) {
                temp[i] /= sum[i];
                temp[i] = max(0.0, min(temp[i], 255.0));
            }
            
            // store back
            data_bf[store_offset + 0] = (uint8_t)temp[0];
            data_bf[store_offset + 1] = (uint8_t)temp[1];
            data_bf[store_offset + 2] = (uint8_t)temp[2];
            if(channels == 4) {
                data_bf[store_offset + 3] = (uint8_t)temp[3];
            }
        }
    }

    bmp_write(ofname, data_bf, file_header, bmp_info_header, false);
    return;
}


void BMP::GrayWorld(double sb_bias = 0, double sg_bias = 0, double sr_bias = 0) {
    // filename.erase(filename.find(".bmp"), 4);
    // string s = filename.substr(filename.find("input")+5);
    // string ofname = "output" + s + "_1.bmp";
    string ofname = ofilename;

    vector<uint8_t> data_gw;
    data_gw.resize(real_image_size);
    int channels = bitsPerPixel / 8;
    int real_row_width = ceil(width * channels / 4.0) * 4;

    int sumR = 0, sumG = 0, sumB = 0;
    double avgR = 0, avgG = 0, avgB = 0;
    double scaleR = 0, scaleG = 0, scaleB = 0;
    double gray = 0;

    // do gray world
    for(int y=0; y<height; ++y) {
        for(int x=0; x<width; ++x) {
            int offset = y * real_row_width + x * channels;
            sumB += (int)data[offset + 0];
            sumG += (int)data[offset + 1];
            sumR += (int)data[offset + 2];
        }
    }

    avgB = sumB / (height * width);
    avgG = sumG / (height * width);
    avgR = sumR / (height * width);
    gray = (avgR + avgG + avgB) / 3.0;
    scaleB = (gray / avgB) + sb_bias;
    scaleG = (gray / avgG) + sg_bias;
    scaleR = (gray / avgR) + sr_bias;

    cout << scaleB << " " << scaleG << " " << scaleR << endl;

    for(int y=0; y<height; ++y) {
        for(int x=0; x<width; ++x) {
            int offset = y * real_row_width + x * channels;
            data_gw[offset + 0] = (uint8_t)min(255.0, ((float)data[offset + 0] * scaleB));
            data_gw[offset + 1] = (uint8_t)min(255.0, ((float)data[offset + 1] * scaleG));
            data_gw[offset + 2] = (uint8_t)min(255.0, ((float)data[offset + 2] * scaleR));
            if(channels == 4) {
                data_gw[offset + 3] = data[offset + 4];
            }
        }
    }

    bmp_write(ofname, data_gw, file_header, bmp_info_header, false);
    return;
}


void BMP::MaxRGB(double sb_bias = 0, double sg_bias = 0, double sr_bias = 0) {
    // filename.erase(filename.find(".bmp"), 4);
    // string s = filename.substr(filename.find("input")+5);
    // string ofname = "output" + s + "_1.bmp";
    string ofname = ofilename;

    vector<uint8_t> data_mrgb;
    data_mrgb.resize(real_image_size);
    int channels = bitsPerPixel / 8;
    int real_row_width = ceil(width * channels / 4.0) * 4;

    int maxR = 0, maxG = 0, maxB = 0, global_max = 0;
    double scaleR = 0, scaleG = 0, scaleB = 0;

    // do max RGB
    for(int y=0; y<height; ++y) {
        for(int x=0; x<width; ++x) {
            int offset = y * real_row_width + x * channels;
            maxB = max(maxB, (int)data[offset + 0]);
            maxG = max(maxG, (int)data[offset + 1]);
            maxR = max(maxR, (int)data[offset + 2]);
            for(int k=0; k<3; ++k) {
                global_max = max(global_max, (int)data[offset + k]);
            }
        }
    }

    scaleB = ((float)maxB / global_max) + sb_bias;
    scaleG = ((float)maxG / global_max) + sg_bias;
    scaleR = ((float)maxR / global_max) + sr_bias;
    
    cout << scaleB << " " << scaleG << " " << scaleR << endl;

    for(int y=0; y<height; ++y) {
        for(int x=0; x<width; ++x) {
            int offset = y * real_row_width + x * channels;
            data_mrgb[offset + 0] = (uint8_t)min(255.0, ((float)data[offset + 0] / scaleB));
            data_mrgb[offset + 1] = (uint8_t)min(255.0, ((float)data[offset + 1] / scaleG));
            data_mrgb[offset + 2] = (uint8_t)min(255.0, ((float)data[offset + 2] / scaleR));
            if(channels == 4) {
                data_mrgb[offset + 3] = data[offset + 4];
            }

        }
    }

    bmp_write(ofname, data_mrgb, file_header, bmp_info_header, false);
    return;
}


int main(int argc, char** argv) {

    // input image file
    string infile = argv[1];
    string outfile = argv[2];

    // get manipulation mode
    string mode = (argc >= 4) ? argv[3] : "-id";

    // for scaling 
    float ratio = 0;
    float beta = 0;
    float kernel_size = 5;
    if(argc == 5) {
        stringstream ss; float f;
        ss << argv[4]; ss >> f;
        ratio = f;
    }
    else if(argc == 6) {
        stringstream ss; float f; float b;
        ss << argv[4]; ss >> f;
        ratio = f;
        ss.clear(); ss << argv[5]; ss >> b;
        beta = b;
    }
    else if(argc == 7) {
        stringstream ss; float f; float b; float k;
        ss << argv[4]; ss >> f;
        ratio = f;
        ss.clear(); ss << argv[5]; ss >> b;
        beta = b;
        ss.clear(); ss << argv[6]; ss >> k;
        kernel_size = k;
    }

    // constructor
    BMP bmp(infile, outfile, mode, ratio, beta, kernel_size);

    // do image processing
    bmp.ImageProcessing();

    return 0;
}


