sum:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 24
    mov [rbp - 32], rdi
    mov [rbp - 40], rsi
    mov rax, [rbp - 32]
    push rax
    mov rax, [rbp - 40]
    mov rbx, rax
    pop rax
    lea rax, [rax + rbx]
    add rsp, 24
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret