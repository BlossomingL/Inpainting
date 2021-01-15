# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author:XLin
# datetime:2020/4/5 0005 10:57
# software: PyCharm
class Solution:
    def numSteps(self, s: str) -> int:
        def s_to_int(s):
            res = 0
            s_len = len(s)
            i = s_len - 1
            while i >= 0:
                res += int(s[i]) * (2 ** (s_len - i - 1))
                i -= 1
            return res

        def int_to_b(num):
            s = []
            while num != 0:
                s.append(str(num % 2))
                num = num // 2
            return ''.join(s[::-1])

        count = 0
        while s != '1':
            if s[-1] == '0':
                v = s_to_int(s) // 2
                s = int_to_b(v)
            elif s[-1] == '1':
                v = s_to_int(s) + 1
                s = int_to_b(v)
            print(s)
            count += 1
        return count
print(Solution().numSteps("1101"))