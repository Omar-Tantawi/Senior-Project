<?php

namespace App\Http\Requests\SalaryPayment;

use Illuminate\Foundation\Http\FormRequest;

class UpdateSalaryPaymentRequest extends FormRequest
{
    public function authorize(): bool
    {
        return true;
    }

    public function rules(): array
    {
        return [
            'amount'       => 'sometimes|numeric|min:0',
            'period_month' => 'sometimes|string',
            'paidat'       => 'sometimes|date',
        ];
    }
}
