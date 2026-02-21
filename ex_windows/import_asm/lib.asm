sum:
    push rbp
    mov rbp, rsp
    push rsi
    push rdi
    push rbx
    push r12
    push r13
    sub rsp, 24
    mov [rbp - 48], rcx
    mov [rbp - 56], rdx
    mov rax, [rbp - 48]
    push rax
    mov rax, [rbp - 56]
    mov rbx, rax
    pop rax
    lea rax, [rax + rbx]
    add rsp, 24
    pop r13
    pop r12
    pop rbx
    pop rdi
    pop rsi
    pop rbp
    ret