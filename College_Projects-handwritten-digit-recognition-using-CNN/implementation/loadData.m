function images = loadData(filename)
    % opening the file 
    fp = fopen(filename,'rb');
    assert(fp ~= -1,['Could not open: ', filename]);
    
    magic = fread(fp,1 ,'int32',0,'ieee-be');
    assert(magic == 2051,['Bad magic number in',filename]);
    
    numImages = fread(fp,1,'int32',0,'ieee-be'); % no of images
    numRows = fread(fp,1,'int32',0,'ieee-be'); % no of rows per image
    numCols = fread(fp,1,'int32',0,'ieee-be'); % no of columns per image
    
    images = fread(fp, inf, 'unsigned char=>unsigned char');
    images = reshape(images, numCols, numRows, numImages);
    images = permute(images,[2 1 3]);
    
    fclose(fp);
    
    % images are reshaped to match the digit's image size
    images = reshape(images, size(images,1)* size(images, 2), size(images,3));
    images = double(images)/255;
    