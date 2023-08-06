# NUEDC 2023 E

> 全国大学生电子设计竞赛 2023年 E题 视觉部分代码

## 硬件:

+ 相机: 大恒相机 (`MER-131-210U3C`)
+ 计算: Jetson Xavier NX
+ 通讯: USB TO TTL

## 软件

+ CUDA (`11.4`)
+ OpenCV (`4.7.0 with CUDA`)
+ DahengSDK (`1.2.2206.9161 2022-06-16`)

## 工作流程

1. 图像处理
   [![](https://mermaid.ink/svg/pako:eNo9kM1Kw0AUhV8l3HUaJpMhP7MQbLvUje5MuhiaqQ00SYkTsIaAiBG6dlNQqAhduBBcqmB9mkl8DCcjeFeXez7uOZwKpnnMgcJ5wZZz4-gkygzjMOwe3tvHz5_9vWx2E2MwODCGobxr5Oum3TzJt9tJjw21MAq77Yv8em63u3a918JIC-Ow_b6Wzbq7-fjXwISUFylLYmVZ9WwEYs5THgFVa8xnrFyICKKsVigrRX66yqZARVFyE8plzAQfJ0yFTYHO2OJCXZcsA1rBJVDPswJCHIxcz_dsFBATVkCx7Vou7idAyAmQi2sTrvJcfUCWjxyCCUI2xsS3AxN4nIi8OP4rRXejHc4038eofwEiPW65?type=png)](https://mermaid-js.github.io/mermaid-live-editor/edit#pako:eNo9kM1Kw0AUhV8l3HUaJpMhP7MQbLvUje5MuhiaqQ00SYkTsIaAiBG6dlNQqAhduBBcqmB9mkl8DCcjeFeXez7uOZwKpnnMgcJ5wZZz4-gkygzjMOwe3tvHz5_9vWx2E2MwODCGobxr5Oum3TzJt9tJjw21MAq77Yv8em63u3a918JIC-Ow_b6Wzbq7-fjXwISUFylLYmVZ9WwEYs5THgFVa8xnrFyICKKsVigrRX66yqZARVFyE8plzAQfJ0yFTYHO2OJCXZcsA1rBJVDPswJCHIxcz_dsFBATVkCx7Vou7idAyAmQi2sTrvJcfUCWjxyCCUI2xsS3AxN4nIi8OP4rRXejHc4038eofwEiPW65)

2. 矩形查找部分
   [![](https://mermaid.ink/svg/pako:eNp1Ul1r2lAY_ivhXKcSNeoxF4W6bm6wMbDdxZZ4kSXHGtATiQlbJ4IXruhWOsugLf0SJFBa2gVGt7V-9c94YvIvdpK0ZSC7O-_zPud9nvejARRdRUAApYr-QSnLhsms5yRct95vGHKtzPiDNrHb896WhBlVM5BiajpmXhZoWDcUMZcvJPKFlbfFpaVlZkV8vb72xj_okNa4yARITpzfjbzJZRQ9EYkzcrtTb_KDjL4XJYyw-o8UcW5nwx3X-Ta7uYgoi5qrone3OxuPiX3oTW_JZBBUdg8ccnQUxYHO03sdsvWZ_Np-TIXU4y5lPBODlL3nXV-TXod0bHfvatFOb-d_nedFv9V1v577J4P5meMet4h9Nu-f32ssM89F98su6fz2T04jmGwPSb8_uxm6h_te92fEeiF6fxwybfv7V9TFg76EH0ceshanEsEP7gALqsioyppKl9ig5hgJmGVURRIQ6FNFJdmqmBKQcJNSZcvU1zaxAgTTsBALrJoqm2hVk2nPVSCU5EqdojUZA6EBPgIhnoaxbCbOJ7gEhBxMZXkWbAKBz8ZoJp6BqRRM8tl0vMmCT7pOK3AxyCX5RJLneS6dhjCTZAFSNVM3XkV3Fp5bKPEu_BD4aP4FzPQl9g?type=png)](https://mermaid-js.github.io/mermaid-live-editor/edit#pako:eNp1Ul1r2lAY_ivhXKcSNeoxF4W6bm6wMbDdxZZ4kSXHGtATiQlbJ4IXruhWOsugLf0SJFBa2gVGt7V-9c94YvIvdpK0ZSC7O-_zPud9nvejARRdRUAApYr-QSnLhsms5yRct95vGHKtzPiDNrHb896WhBlVM5BiajpmXhZoWDcUMZcvJPKFlbfFpaVlZkV8vb72xj_okNa4yARITpzfjbzJZRQ9EYkzcrtTb_KDjL4XJYyw-o8UcW5nwx3X-Ta7uYgoi5qrone3OxuPiX3oTW_JZBBUdg8ccnQUxYHO03sdsvWZ_Np-TIXU4y5lPBODlL3nXV-TXod0bHfvatFOb-d_nedFv9V1v577J4P5meMet4h9Nu-f32ssM89F98su6fz2T04jmGwPSb8_uxm6h_te92fEeiF6fxwybfv7V9TFg76EH0ceshanEsEP7gALqsioyppKl9ig5hgJmGVURRIQ6FNFJdmqmBKQcJNSZcvU1zaxAgTTsBALrJoqm2hVk2nPVSCU5EqdojUZA6EBPgIhnoaxbCbOJ7gEhBxMZXkWbAKBz8ZoJp6BqRRM8tl0vMmCT7pOK3AxyCX5RJLneS6dhjCTZAFSNVM3XkV3Fp5bKPEu_BD4aP4FzPQl9g)

3. 光点查找部分
   [![](https://mermaid.ink/svg/pako:eNplkktLw0AQx79KmHNb2jxMmoNgrY-DXnwdTHpYk60NtJuSJvgohSoqVlEERcQHQhFURAueFK2fpmncb-GmqaL2tvuf38z8Z3eqYNgmBhXyRXvFKCDH5eYyOql4S8sOKhc42tzyb7aCox2dcKblYMO1bMJNzbBrxTG0zMQMPzm7kIvHh7kRbXQ-O-K33rqND3-7EWy-5rhQH9Mi7bP95L8d53SCifmrw_-kwUYZLbi-99tNuvEUtO_93Zvu6WMu1Fl5__IuCnZe9lizUW2B1s_pxnGnfUjPdv36-zfYeTn4x2a_Wfpw9sP-9RY8XtB6Y8DRuEavmsFtKwqHw09o3dZJ52O_r_TmntRo8_Kz8dzX-rV18vOkITU4_e9wfDhKhhiUsFNClsl-qsr8cDq4BVzCOqjsaOI88oquDjqpMRR5rj27RgxQXcfDMfDKJnJx1kJsqBKoeVSsMLWMCKhVWAU1NaQkhkReSSuSlJbkpBCDNVDFdEJMpURFluS0IEpyLQbrts3ykwklKYi8IIgCz8sMxqbl2s50tEe9depVX-zRoYXaF60GDa0?type=png)](https://mermaid-js.github.io/mermaid-live-editor/edit#pako:eNplkktLw0AQx79KmHNb2jxMmoNgrY-DXnwdTHpYk60NtJuSJvgohSoqVlEERcQHQhFURAueFK2fpmncb-GmqaL2tvuf38z8Z3eqYNgmBhXyRXvFKCDH5eYyOql4S8sOKhc42tzyb7aCox2dcKblYMO1bMJNzbBrxTG0zMQMPzm7kIvHh7kRbXQ-O-K33rqND3-7EWy-5rhQH9Mi7bP95L8d53SCifmrw_-kwUYZLbi-99tNuvEUtO_93Zvu6WMu1Fl5__IuCnZe9lizUW2B1s_pxnGnfUjPdv36-zfYeTn4x2a_Wfpw9sP-9RY8XtB6Y8DRuEavmsFtKwqHw09o3dZJ52O_r_TmntRo8_Kz8dzX-rV18vOkITU4_e9wfDhKhhiUsFNClsl-qsr8cDq4BVzCOqjsaOI88oquDjqpMRR5rj27RgxQXcfDMfDKJnJx1kJsqBKoeVSsMLWMCKhVWAU1NaQkhkReSSuSlJbkpBCDNVDFdEJMpURFluS0IEpyLQbrts3ykwklKYi8IIgCz8sMxqbl2s50tEe9depVX-zRoYXaF60GDa0)

4. 任务调度
   [![](https://mermaid.ink/svg/pako:eNolkM1OwkAUhV9lctflr7bMtAsTEeNGN-DKDouxHaQJnSFlGkVCIkujCbrBhSvjQncYowRDfBta4S0c6e6eL-fee3KG4MuAgwvtrrzwOyxW6KRGRT85O49Zr4NWi2X2Nf59u6UCBWHMfRVKgY4aWu5569kynUxXi4908tJChcIuqnnZ02c2fW-UDkvNHO172c3r-vkuV3UvXc6zx3k6-9783OfswEsnD5vrcYsKLgIwIOJxxMJApxrqP4iC6vCIU3D1GPA2S7qKAhUjbWWJks2B8MFVccINSHoBU7weMh0_ArfNun1Ne0yAO4RLcCsWLppOBVukbFUJtg0YaGjbxbJdJTvEIY5pYzIy4EpKvV8uYuyYDjEtxzRxxaoawINQyfg4b21b3vb-6db_H2L0BxJde_0?type=png)](https://mermaid-js.github.io/mermaid-live-editor/edit#pako:eNolkM1OwkAUhV9lctflr7bMtAsTEeNGN-DKDouxHaQJnSFlGkVCIkujCbrBhSvjQncYowRDfBta4S0c6e6eL-fee3KG4MuAgwvtrrzwOyxW6KRGRT85O49Zr4NWi2X2Nf59u6UCBWHMfRVKgY4aWu5569kynUxXi4908tJChcIuqnnZ02c2fW-UDkvNHO172c3r-vkuV3UvXc6zx3k6-9783OfswEsnD5vrcYsKLgIwIOJxxMJApxrqP4iC6vCIU3D1GPA2S7qKAhUjbWWJks2B8MFVccINSHoBU7weMh0_ArfNun1Ne0yAO4RLcCsWLppOBVukbFUJtg0YaGjbxbJdJTvEIY5pYzIy4EpKvV8uYuyYDjEtxzRxxaoawINQyfg4b21b3vb-6db_H2L0BxJde_0)
   [![](https://mermaid.ink/svg/pako:eNpdkD1PwlAUhv_KzRmMxpa0hdKPwcEYERIXdZLLcKUXIaG3pLRRJCzGRFAJky44aCTBhY_FiA7wZ2jpz_AC0cHt5r3Pk3POW4e8Y1EwoVB2LvJF4nroZBezqn927pJKEYWtfvT6cEQtzPY3qx6tbCFRDFvToDledIfhy-3i-ksUd1AqG32OgulN0OyFT4McWmaZbDC-WXzPwo_7aPSYwyzF4-D5fU3OJ3ccOuBMZz4bhoO37aDXD1rtaNblaGb5hzZQem3NJ-1_YjrL8V9x3PkTKbNAAJu6NilZ_K46Zghh8IrUphhM_rRogfhlDwNmDY4S33OOaywPpuf6VAC_YhGP7pUIL8AGs0DKVZ5WCAOzDpdgygktphiyltClRFLXVAFqPFTVmKQm9bhu6IaianpDgCvH4b4U0zRDMXRF0WVZUuKSIQC1Sp7jHq6LX_W_GnC6EpZbNH4AxPmnHQ?type=png)](https://mermaid-js.github.io/mermaid-live-editor/edit#pako:eNpdkD1PwlAUhv_KzRmMxpa0hdKPwcEYERIXdZLLcKUXIaG3pLRRJCzGRFAJky44aCTBhY_FiA7wZ2jpz_AC0cHt5r3Pk3POW4e8Y1EwoVB2LvJF4nroZBezqn927pJKEYWtfvT6cEQtzPY3qx6tbCFRDFvToDledIfhy-3i-ksUd1AqG32OgulN0OyFT4McWmaZbDC-WXzPwo_7aPSYwyzF4-D5fU3OJ3ccOuBMZz4bhoO37aDXD1rtaNblaGb5hzZQem3NJ-1_YjrL8V9x3PkTKbNAAJu6NilZ_K46Zghh8IrUphhM_rRogfhlDwNmDY4S33OOaywPpuf6VAC_YhGP7pUIL8AGs0DKVZ5WCAOzDpdgygktphiyltClRFLXVAFqPFTVmKQm9bhu6IaianpDgCvH4b4U0zRDMXRF0WVZUuKSIQC1Sp7jHq6LX_W_GnC6EpZbNH4AxPmnHQ)
   R = Red(`发射红色激光的机器人`) ; G = Green(`发射绿色激光的机器人`) ; S = Stop(`停止所有任务`)
