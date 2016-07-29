#echo 'Learning Rates'
#nohup python LSTM_Clfn_finetuned.py --gpu 0  --output-weights ../Output/ModelParams/LSTM_params_finetuned_WE_5E.h5  -R 5 --vectors ../Finetuned_Embeddings/epochs/word2Ind_dict_5E.pkl --pretrained ../Finetuned_Embeddings/epochs/wordEmbedMat_5E.pkl &
#nohup python LSTM_Clfn_finetuned.py --gpu 1  --output-weights ../Output/ModelParams/LSTM_params_finetuned_WE_1E.h5  -R 5 --vectors ../Finetuned_Embeddings/epochs/word2Ind_dict_1E.pkl --pretrained ../Finetuned_Embeddings/epochs/wordEmbedMat_1E.pkl &
#nohup python LSTM_Clfn_finetuned.py --gpu 3  --output-weights ../Output/ModelParams/LSTM_params_finetuned_WE_3E.h5  -R 5 --vectors ../Finetuned_Embeddings/epochs/word2Ind_dict_3E.pkl --pretrained ../Finetuned_Embeddings/epochs/wordEmbedMat_3E.pkl &

#nohup python LSTM_Clfn_finetuned.py --gpu 3  --output-weights ../Output/ModelParams/Question_label/LSTM_params_finetuned_7WL.h5  -R 3 --vectors ../Finetuned_Embeddings/window_length/word2Ind_dict_7WL.pkl --pretrained ../Finetuned_Embeddings/window_length/wordEmbedMat_7WL.pkl --main-info-file '../Output/results/LSTM_results_Question_label.txt' &


#nohup python LSTM_Bidirectional.py --gpu 2  --output-weights ../Output/ModelParams/Question_label/LSTM_params_BiDi_7WL.h5  -R 3 --vectors ../Finetuned_Embeddings/window_length/word2Ind_dict_7WL.pkl --pretrained ../Finetuned_Embeddings/window_length/wordEmbedMat_7WL.pkl --main-info-file '../Output/results/LSTM_BiDi_Question_label.txt' &


python -i LSTM_Bidirectional.py --gpu 3  --model-weights ../Output/ModelParams/LSTM_params_BiDi_2L_128_64.h5


#echo 'Neurons in layer'

#nohup python LSTM_Clfn_finetuned.py --gpu 1 --learning-rate 0.0001 --output-weights ../Output/ModelParams/LSTM_params_finetuned_1L_128.h5 --neurons '128'  -R 5 & 
#nohup python LSTM_Clfn_finetuned.py --gpu 3 --learning-rate 0.0001 --neurons '64' --output-weights ../Output/ModelParams/LSTM_params_finetuned_1L_64.h5 -R 5 & 
#nohup python LSTM_Clfn_finetuned.py --gpu 0 --learning-rate 0.0001 --neurons '256' --output-weights ../Output/ModelParams/LSTM_params_finetuned_1L_256.h5 -R 5 & 
#
#echo '1 Layer variations'
#nohup python LSTM_Bidirectional.py --gpu 0 --learning-rate 0.001 --neurons '64' --num-epochs 10 --num-layers 1 --output-weights ../Output/ModelParams/LSTM_params_BiDi_1L_64.h5 -R 3 & 
#
#nohup python LSTM_Bidirectional.py --gpu 1 --learning-rate 0.001 --neurons '128' --num-epochs 10 --num-layers 1 --output-weights ../Output/ModelParams/LSTM_params_BiDi_1L_128.h5 -R 3 & 
#
#echo '2-Layer variations'
#nohup python LSTM_Bidirectional.py --gpu 2 --learning-rate 0.001 --neurons '64,32' --num-epochs 10 --num-layers 2 --output-weights ../Output/ModelParams/LSTM_params_BiDi_2L_64_32.h5 -R 3 & 
#
#nohup python LSTM_Bidirectional.py --gpu 3 --learning-rate 0.001 --neurons '128,64' --num-epochs 10 --num-layers 2 --output-weights ../Output/ModelParams/LSTM_params_BiDi_2L_128_64.h5 -R 3 & 
#

