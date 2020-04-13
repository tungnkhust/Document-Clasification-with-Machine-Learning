# Course Project Machine Learning
## Introduction
Đây là bài tập lớn học phần IT3190 - Nhập môn học máy và khai phá dữ liệu Viện Công nghệ thông tin và truyền thông Đại học Bách Khoa Hà Nội.

Thành viên nhóm:
1. Nguyễn Văn Trung     20173424    Viện Công nghệ thông tin và truyền thông Đại học Bách Khoa Hà Nội
2. Bùi Minh Tuấn        20173444    Viện Công nghệ thông tin và truyền thông Đại học Bách Khoa Hà Nội
3. Nguyễn Hữu Anh Việt  20173465    Viện Công nghệ thông tin và truyền thông Đại học Bách Khoa Hà Nội
4. Nguyễn Kỳ Tùng       20173455    Viện Công nghệ thông tin và truyền thông Đại học Bách Khoa Hà Nội

Đề tài: Phần loại văn bản tiếng anh
Giới thiệu đề tài:

## Dữ liệu
Bộ dữ liệu sử dụng là bộ data Reuters10 được tách ra từ bộ data bộ dữ liệu Reuters-21578 từ trang  Martin Thoma: https://martin-thoma.com/nlp-reuters/
Bộ dữ liệu gốc gồm 90 classes, 7769 training documents, 3019 testing documents.
Do dữ liệu gốc phân bố không đều và nên chúng tôi đã tách ra thành bộ Reuters 10 với 10 lớp có số lượng sample lơn nhất:
earn, acq, money-fx, grain, crude, trade, interest, wheat, ship, corn

Chi tiết bộ Reuters 10:
Gồm 7193 training documents, 2787 testing documents.
Dữ liệu được chia thành 2 folders train và test.
Trong mỗi folder sẽ được tổ chức thành 10 folders con.
Mỗi folder sẽ chứa các samples dưới dạng file .txt tương ứng với class đó.
Thống kế dữ liệu:
Class_name      num_of_documents(train)     num_of_documents(test)      mean_num_words_in_train
earn            2877
acq             1650
mony-fx         538
grain           433
crude           389
trade           369
interest        347
wheat           212
ship            197
corn            187

Lưu trữ: https://drive.google.com/open?id=1BbwomSsHt0bdIyH_iFJXGtDNmw875BZ3

## Cài đặt môi trường

## Chạy thử nhiệm

train model:

test model:

inference: