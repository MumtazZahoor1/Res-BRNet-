
function Iout=preprocess_malaria_images(filename,desired_size)

% This function preprocesses malaria images using color constancy
% technique and later reshapes them to an image of desired size
% Author: Barath Narayanan

% Read the Image
I=imread(filename);

% Some images might be grayscale, replicate the image 3 times to
% create an RGB image.
% 
if ismatrix(I)
    I=cat(3,I,I,I);
end

% Conversion to Double for calculation purposes
I=double(I);
% % 
% % % % % Mean Calculation
Ir=I(:,:,1);mu_red=mean(Ir(:));
Ig=I(:,:,2);mu_green=mean(Ig(:));
Ib=I(:,:,3);mu_blue=mean(Ib(:));
mean_value=(mu_red+mu_green+mu_blue)/3;

% Scaling the Image for Color constancy
Iout(:,:,1)=I(:,:,1)*mean_value/mu_red;
Iout(:,:,2)=I(:,:,2)*mean_value/mu_green;
Iout(:,:,3)=I(:,:,3)*mean_value/mu_blue;

% Converting it back to uint8
Iout=uint8(I);

% Resize the image
Iout=imresize(Iout,[desired_size(1) desired_size(2)]);
end


