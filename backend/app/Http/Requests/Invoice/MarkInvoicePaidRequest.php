<?php

namespace App\Http\Requests\Invoice;

use Illuminate\Foundation\Http\FormRequest;

class MarkInvoicePaidRequest extends FormRequest
{
    public function authorize(): bool
    {
        return true;
    }

    public function rules(): array
    {
        return [
            'method'      => 'nullable|string|in:cash,card,bank_transfer,cheque',
            'guardian_id' => 'nullable|integer',
        ];
    }
}
