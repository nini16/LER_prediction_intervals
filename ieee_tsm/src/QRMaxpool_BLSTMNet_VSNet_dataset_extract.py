# ---------- Details ---------------
# Main thing to note here is the order of inputs
# (left-edgenet_ler, left-normcp_approach2, left-normcp_natlog5, right-edgenet_ler, right-normcp_approach2, right-normcp_natlog5, noise_param, noise_param)
# noise_param: We run estimate_sigma on a noisy image. (https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.estimate_sigma)
# notice that the noisy image is not normalized!! The values in the noisy image are not divided by 256, hence why sigma turns out quite large
# also the noise param is repeated because we use the same value for the same image, duh



dataset = np.zeros((89280, 8))

sigmas = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Xis = [6,7,8,9,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,38,39]
widths = [20, 30]
noises = [2, 3, 4, 5, 10, 20, 30, 50, 100, 200]

print('###################### Begin Extraction ###############################')

count = 0
for sigma in sigmas:
    for alpha in alphas:
        print("working on alpha: {}".format(alpha))
        for Xi in Xis:
            for width in widths:
                for s in range(2):
                    for noise in noises:
                        space = math.floor(width*2**s)
                        shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16)
                        
                        noisy_file = 'nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'
                        
                        noise_value = dic[noisy_file]
                        dataset[count][:6] = X_train_QR_5[count][:]
                        dataset[count][6] = noise_value
                        dataset[count][7] = noise_value
                        if (count%9920==1): print("Current input value is: {}".format(dataset[count]))
                        
                        count += 1
    print("finished sigma: {}".format(sigma))
    print("count: {}".format(count))

np.save('/scratch/user/nini16/IEEE_TSM/X_train_QRMaxpool_BLSTMNet_VSNet_train.npy', dataset)