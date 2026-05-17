<?php

namespace App\Http\Requests\SalaryPayment;

use Illuminate\Foundation\Http\FormRequest;

class StoreSalaryPaymentRequest extends FormRequest
{
    public function authorize(): bool
    {
        return true;
    }

    public function rules(): array
    {
        return [
            'teacher_id'   => 'required|exists:teachers,id',
            'amount'       => 'required|numeric|min:0',
            'period_month' => 'required|string',
            'paidat'       => 'nullable|date',
        ];
    }
}
