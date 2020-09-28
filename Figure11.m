% This code generates Figure 11, that describes the impact of percentage of
% training and testing images on the computed RMSE.
%% Please run the individual selections separately

clc;
no_of_exp = 100;


%%
% Training images
filename = './cubicbox/training.xlsx';
excel_data=xlsread(filename);
RMSE_vector = excel_data(:,3);
data_length = length(RMSE_vector);
datamatrix = reshape(RMSE_vector,[no_of_exp,data_length/no_of_exp]);
datamatrix(:,20)=[];
figure('Position', [400, 250, 330, 290]);
boxplot(datamatrix,'Whisker',5,'Labels',{'','10','','20','','30','','40','','50','','60','','70','','80','','90',''});
grid on;
box on ;
ylim([165 190]);
xlabel('Percentage of training images [%]','FontSize',12); hold on;
ylabel('RMSE of training images [W/m^{2}]','FontSize',12); hold on;
set(gca,'fontsize',12);

%%


% Testing images
filename = './cubicbox/testing.xlsx';
excel_data=xlsread(filename);
RMSE_vector = excel_data(:,3);
data_length = length(RMSE_vector);
datamatrix = reshape(RMSE_vector,[no_of_exp,data_length/no_of_exp]);
datamatrix(:,20)=[];
figure('Position', [400, 250, 330, 290]);
boxplot(datamatrix,'Whisker',5,'Labels',{'','10','','20','','30','','40','','50','','60','','70','','80','','90',''});
grid on;
box on ;
ylim([165 190]);
xlabel('Percentage of testing images [%]','FontSize',12); hold on;
ylabel('RMSE of testing images [W/m^{2}]','FontSize',12); hold on;
set(gca,'fontsize',12);       

%%