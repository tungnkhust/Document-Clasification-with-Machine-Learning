# Course Project Machine Learning
Đây là bài tập lớn học phần IT3190 - Nhập môn học máy và khai phá dữ liệu Viện Công nghệ thông tin và truyền thông Đại học Bách Khoa Hà Nội.

Thành viên nhóm:

|Họ và tên | Mssv|
|--- | ---|
|Nguyễn Văn Trung	       |20173424|
|Bùi Minh Tuấn	           |20173444|
|Nguyễn Hữu Anh Việt	   |20173465|
|Nguyễn Kỳ Tùng	           |20173455|

## Đề tài: Phần loại văn bản tiếng anh
Giới thiệu đề tài:
Phân loại văn bản là một bài toán điển hình trong lĩnh vực xử lý ngôn ngữ tự nhiên.
Đây là một bài toán hay với độ phức tạp khác nhau phụ thuộc vào từng chủ đề riêng của bài toán và loại văn bản cần phân loại.
Hiện tại đã có rất nhiều hướng tiếp cận như sử dụng các giải thuật học máy truyền thống (Naive Bayes, SVM) trong phân loại hay các kĩ thuật hiện đại hơn như học sâu đều cho những kết quả rất khả quan.
Trong khuôn khổ của môn học chúng tôi đề xuất tìm hiểu kỹ thuật học máy SVM trong phân loại bài báo Reuters.

## Dữ liệu
Bộ dữ liệu sử dụng là bộ data Reuters10 được tách ra từ bộ data bộ dữ liệu Reuters-21578 từ trang  Martin Thoma: https://martin-thoma.com/nlp-reuters/
Bộ dữ liệu gốc gồm 90 classes, 7769 training documents, 3019 testing documents.
Do dữ liệu gốc phân bố không đều và nên chúng tôi đã tách ra thành bộ Reuters 10 với 10 lớp có số lượng sample lơn nhất:
earn, acq, money-fx, grain, crude, trade, interest, wheat, ship, corn.

Chi tiết bộ Reuters 10:
Gồm 7193 training documents, 2787 testing documents.
Dữ liệu được chia thành 2 folders train và test.
Trong mỗi folder sẽ được tổ chức thành 10 folders con.
Mỗi folder sẽ chứa các samples dưới dạng file .txt tương ứng với class đó.

Download dữ liệu từ link sau và lưu trong thư mục data:
```
https://drive.google.com/file/d/1ViLpFpalxgxMVsf2HXvVvl1Fd2i_qnMw/view?usp=sharing
```

## Cài đặt môi trường
Nếu bạn không có môi trường ảo anaconda, bạn có thể cài đặt theo hưỡng dẫn sau:
- [Install Anaconda3 on Ubuntu 18.04](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)

Sau đó bạn tạo môi trường ảo anaconda mới với tên "pj2":<br>
`conda create --name pj2 python=3.7`

Bạn kích hoạt môi trường mới:<br>
`conda activate pj2`

Cuối cùng bạn cài các thư viện cần thiết để chạy Project
```
pip install -r requirements.txt
```
## Mô hình đề xuất
Do phận vi của môn học, chúng tôi xin đề xuất mô hình học máy SVM với việc sử dụng kỹ thuật tf-idf để trích xuất đặc trưng.

![Mô hình](https://drive.google.com/file/d/1eXF66ReE1yYaeQChAbGnJwmo7CgAGEhu/view?usp=sharing)

## Kết quả
|Accuracy | F1 Score| Precision | Recall|
|--- | ---| ---| ---|
|0.8873 |0.7085 | 0.7272| 0.7034|

Chi tiết về giải thuật và kết quả thực nghiệm [xem tại đây](https://docs.google.com/document/d/1Vt8y_zFxWrU_7HJq458iqukgXmM82HNTs8Mu0iuhesk/edit?usp=sharing)

## Chạy thử nghiệm
Xử lý dữ liệu:
```
python preprocessing.py --train_dir data/Reuter10/train \
--test_dir data/Reuter10/test \
--cutoff 26 \
--hier True
```

train_dir và test_dir là đường dẫn đến thư mục chưa dữ liệu trên và test mà khi tải dư liệu về lưu tại đó.
cutoff là thông số để loại bỏ các từ xuất hiện không quá n lần, mặc định cutoff=0.
hier là biến để xem có xử lý dữ liệu cho mô hình học phân cấp hay không, mặc định là False
Có thể chạy mặc định với lệnh sau:

```
python preprocessing.py
```
Huấn luyện và đánh giá model:
```
python train.py --train_file data/full_data/data.csv \
--test_file data/full_data/test.csv \
--vocab_file vocab/vocab.csv \
--stopword_file vocab/stopword.txt \
--sublinear_tf True \
--kernel 'linear' \
--C 1 \
--hier True \
--seed 1337 \
--show_cm_matrix True
```
vocab_file là danh sách các từ dùng làm từ điển, mặc định là "".
stopword_file là danh sách các từ stopword, mặc định là "". 
kernel là kernel dùng trong giải thuật svm, nếu kernel là linear thì mô hình sử dụng LinearSCV() nếu không sẽ sử dụng SVC() với kernel tương ứng, mặc định là "linear".
C là thông số điều chỉnh độ chịu lỗi của giải thuật svm, mặc định là 1.
hier=True thì dùng mô hình phân cấp, mặc định là False.
show_cm_matrix=True thì sẽ hiện thị 2 confusion matrix (nomalize, non-nomalize) sau khi chạy xong kết quả lưu trong thư mục results, mặc định là False.
Chạy mặc định với lệnh sau:
```
python train.py
```

