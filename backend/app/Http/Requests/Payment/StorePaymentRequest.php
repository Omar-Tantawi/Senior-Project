<?php

namespace App\Http\Requests\Payment;

use Illuminate\Foundation\Http\FormRequest;

class StorePaymentRequest extends FormRequest
{
    public function authorize(): bool
    {
        return true;
    }

    public function rules(): array
    {
        return [
            'invoice_id' => 'required|exists:invoice,invoice_id',
            'parent_id'  => 'required|exists:parent,parent_id',
            'amount'     => 'required|numeric|min:0.01',
            'method'     => 'required|in:cash,card,bank_transfer,cheque,stripe',
            'paidat'     => 'nullable|date',
        ];
    }
}
