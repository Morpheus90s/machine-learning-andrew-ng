def gradient_descent(x, y, w, b, alpha, iterations):
    m = len(x)

    for _ in range(iterations):
        dj_dw = 0
        dj_db = 0

        for i in range(m):
            prediction = w * x[i] + b
            dj_dw += (prediction - y[i]) * x[i]
            dj_db += (prediction - y[i])

        dj_dw /= m
        dj_db /= m

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

    return w, b
