function snrs = estimate_snr(lt_typical,lt_std, radiance)
%计算SNR，输入是每个波段的参考辐亮度，参考辐亮度的标准差和，radiance
% lt_typical unit: mW cm−2 μm−1 sr−1
% radiance: uw m-3 um-1 sr-1

radiance = radiance .* 10;% unit convert

lowest_radiance = lt_typical - lt_std;
highest_radiance = lt_typical + lt_std;
radiance(radiance < lowest_radiance) = NaN;
radiance(radiance > highest_radiance) = NaN;

lines = size(radiance,1);
cols = size(radiance,2);
snrs = [];
N = 1;
fprintf('Process:');
for i=  2:lines - 1
    for j= 2:cols - 1
        window = radiance(i-1:i+1, j-1:j+1);        
        if numel(find(isnan(window))) < 1
            ratio = nanmax(window(:))/nanmin(window(:));
            if ratio <= 1.005
                win_std = nanstd(window(:));
                if win_std>0
                    snr = nanmean(window(:))./win_std;
                    snrs = [snrs;snr];
                end
            end
        end        
    end    
    %print process to screen
    if mod(i,fix(lines/10)) == 0
        fprintf('%d...',N*10);
        N = N + 1;
    end
end
fprintf('\n');
end